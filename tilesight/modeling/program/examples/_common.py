"""Helpers shared by the example programs (tile counts, causal masks, launch orders, a small reporter)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..contract import ExplicitOrder, WorkGroup, WorkRun, WorkUnits, dispatch_coordinates

TILELANG_COMMIT = "4c9cf5c7ea485e62312fb205c01cec68ad836075"


def _ceil(a: int, b: int) -> int:
    return -(-a // b)


# ---------------------------------------------------------------------------
# Tail / masking helpers
# ---------------------------------------------------------------------------


def causal_kv_trips(q_tile: int, block_m: int, block_n: int, kv_tiles: int) -> int:
    """TileLang: ``min(ceildiv(seq, block_N), ceildiv((bx + 1) * block_M, block_N))``."""

    return min(kv_tiles, _ceil((q_tile + 1) * block_m, block_n))


def causal_valid_fraction(q_tile: int, block_m: int, block_n: int, trips: int) -> float:
    """Exact unmasked fraction of the ``trips`` score tiles of one causal q tile."""

    return causal_valid_fraction_rows(q_tile * block_m, block_m, block_n, trips)


def causal_valid_fraction_rows(first_row: int, rows: int, block_n: int, trips: int) -> float:
    """Unmasked fraction of the ``rows x trips*block_n`` scores of q rows ``first_row ..``."""

    kv_end = trips * block_n
    valid = sum(min(row + 1, kv_end) for row in range(first_row, first_row + rows))
    return valid / float(rows * trips * block_n)


def sectioned_lpt_order(extents: Tuple[int, ...], section: int) -> ExplicitOrder:
    """Launch order of the flash-attention SM100 static grid: head/batch coordinates in
    sections of ``section`` (the L2 swizzle), and inside a section the q blocks from last
    to first, so the longest causal blocks are issued first (``q`` is the last work axis)."""

    heads = dispatch_coordinates(tuple(extents[:-1]), "linear_block_id")
    order = []  # type: List[Tuple[int, ...]]
    for begin in range(0, len(heads), max(section, 1)):
        for q in range(extents[-1] - 1, -1, -1):
            order.extend(head + (q,) for head in heads[begin:begin + max(section, 1)])
    return ExplicitOrder(tuple(order), "sectioned_lpt_%d" % section)


def _attention_work_units(
    q_tiles: int, per_q_units: int, block_m: int, block_n: int, kv_tiles: int, causal: bool,
    kv_valid_fraction: float, masked_ops: Tuple[str, ...], kv_payload_ops: Tuple[str, ...],
    q_order: Optional[List[int]] = None,
    op_rows: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Optional[WorkUnits]:
    """Work groups in real grid order (``q`` is the fastest axis of the launch).

    A causal q tile gets its own group: trip count ``ceildiv((bx+1)*bm, bn)`` and
    the exact unmasked fraction of its score tiles.  Two q tiles share a group
    only when both the trip count and the mask fraction are identical.
    ``op_rows`` gives ``op -> (row offset inside the q tile, rows)`` for ops that
    work on a sub-range of the q tile (the Q stages of one work unit); their mask
    fraction is that of their own rows, not the tile average.
    """

    if not causal:
        if kv_valid_fraction >= 1.0:
            return None
        fractions = {op: kv_valid_fraction for op in set(masked_ops) | set(kv_payload_ops)}
        return WorkUnits((("kv_tail", WorkGroup(effective_fractions=fractions)),),
                         (WorkRun("kv_tail", q_tiles * per_q_units),))
    groups = {}  # type: Dict[Tuple[int, float], Tuple[str, WorkGroup]]
    by_q = []  # type: List[str]
    for index in range(q_tiles):
        trips = causal_kv_trips(index, block_m, block_n, kv_tiles)
        fractions = {}
        for op in masked_ops:
            offset, rows = (op_rows or {}).get(op, (0, block_m))
            fractions[op] = causal_valid_fraction_rows(index * block_m + offset, rows, block_n, trips)
        key = (trips, tuple(round(fractions[op], 12) for op in masked_ops))
        if key not in groups:
            groups[key] = ("q%d_kv%d" % (index, trips), WorkGroup(
                loop_trip_counts={"kv": trips}, effective_fractions=fractions))
        by_q.append(groups[key][0])
    order = []  # type: List[WorkRun]
    if q_order is None:                   # linear block id: q fastest, then hg, kv_head, batch
        q_order = [index for _unit in range(per_q_units) for index in range(q_tiles)]
    for index in q_order:
        gid = by_q[index]
        if order and order[-1].group_id == gid:
            order[-1] = WorkRun(gid, order[-1].count + 1)
        else:
            order.append(WorkRun(gid, 1))
    used = {run.group_id for run in order}
    return WorkUnits(tuple(item for item in groups.values() if item[0] in used), tuple(order))


def _attention_shapes(seq_len: int, block_m: int, block_n: int, partial_q: bool = False) -> Tuple[int, int, float]:
    if seq_len % block_m and not partial_q:
        raise ValueError("seq_len=%d must be a multiple of block_m=%d (partial causal q tiles are not modelled)" % (seq_len, block_m))
    q_tiles = _ceil(seq_len, block_m)
    kv_tiles = _ceil(seq_len, block_n)
    kv_valid = seq_len / float(kv_tiles * block_n)
    return q_tiles, kv_tiles, kv_valid


def report(result: Any, loop: Optional[str] = None, op: Optional[str] = None, max_assumptions: int = 8) -> None:
    """Print what an example run is usually read for: times, II bounds, one op's traffic, and the assumptions.

    Unknown values stay visible: a traffic field or share that the model does not
    know is printed as ``unknown`` with its reason, never dropped like a zero.
    """

    print("program total: %.3f us" % (result.program.total_s * 1e6))
    for name, launch in result.launches.items():
        print("  launch %-8s body=%.3f us  launch=%.3f us  host=%.3f us  work units=%d on %d slots (%d active SMs, residency %s: %s)"
              % (name, launch.kernel_body_s * 1e6, launch.launch_overhead_s * 1e6, launch.host_dispatch_s * 1e6,
                 launch.work_units, launch.slots, launch.active_sms, launch.residency, launch.residency_source))
    if loop is not None:
        for group in result.regions[loop].groups:
            ii = group.ii
            if ii is None:
                continue
            print("  %s group %-10s trips=%-4s II=%.2f ns (resource %.2f, recurrence %.2f, credit %.2f) %s complete=%s"
                  % (loop, group.work_group_id, group.trip_count, ii.selected_ii_s * 1e9, ii.resource_ii_s * 1e9,
                     ii.recurrence_ii_s * 1e9, ii.credit_ii_s * 1e9, ii.scope, ii.search_complete))
    if op is not None:
        item = result.ops[op]
        print("  %s traffic (level, direction: cumulative bytes, share of launch):" % op)
        hidden = 0
        for level, direction, amount, share in item.breakdown(scope="launch"):
            field_share = item.share("%s_%s_bytes" % (level, direction), "launch") or item.share(
                "%s_%s_request_bytes" % (level, direction), "launch")
            reason = getattr(field_share, "reason", "") or "not reported"
            if amount == 0.0 and share in (0.0, None):
                hidden += 1                                   # a known zero, not an unknown
                continue
            print("    %-8s %-5s: %s, %s" % (
                level, direction, "unknown" if amount is None else "%.0f B" % amount,
                ("unknown (%s)" % reason) if share is None else "%.1f%%" % (100.0 * share)))
        if hidden:
            print("    (%d level/direction fields are exactly zero)" % hidden)
    for report_item in result.cache:
        print("  cache model: %s, exact=%s, sampled %s of %s requests" % (
            report_item.reduction_fidelity, report_item.exact, report_item.sampled_requests, report_item.represented_requests))
    diagnostics = result.diagnostics
    for text in diagnostics.unsupported:
        print("  unsupported:", text)
    assumptions = list(dict.fromkeys(diagnostics.approximations))
    if assumptions:
        print("  assumptions and approximations (%d):" % len(assumptions))
        for text in assumptions[:max_assumptions]:
            print("    -", text)
        if len(assumptions) > max_assumptions:
            print("    ... %d more in result.diagnostics.approximations" % (len(assumptions) - max_assumptions))
    slots = [note for note in diagnostics.notes if "effective on-chip slots" in note or "declared " in note]
    for text in slots:
        print("  storage:", text)
