"""Export an ``AnalysisResult`` as Perfetto-loadable Chrome Trace JSON.

Model times are seconds; the trace uses microseconds.  Tracks:

* process ``<launch> spatial`` – one thread per SM slot with work units
  (only when the launch recorded a complete unit timeline);
* process ``<launch> work unit [<group>]`` – one representative work unit:
  resource service intervals on ``res:<resource>`` threads and op
  completion intervals on ``actor:<actor>`` threads.  Repeated periods are
  drawn up to ``max_periods`` times and then summarised by a marker carrying
  ``count``/``stride_s``/``omitted`` so no fake full timeline is produced.
* entries without a known start (no schedule witness) are emitted only as
  summary instants with cumulative service, never as timed bars.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .results import AnalysisResult, RegionGroupResult

US = 1.0e6


def _instant(name: str, pid: int, tid: int, ts: float, args: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "ph": "i", "s": "t", "pid": pid, "tid": tid, "ts": ts * US, "args": args}


def _complete(name: str, pid: int, tid: int, ts: float, dur: float, args: Dict[str, Any], cat: str) -> Dict[str, Any]:
    return {"name": name, "ph": "X", "cat": cat, "pid": pid, "tid": tid, "ts": ts * US, "dur": dur * US, "args": args}


def _meta(pid: int, tid: Optional[int], name: str) -> Dict[str, Any]:
    if tid is None:
        return {"name": "process_name", "ph": "M", "pid": pid, "args": {"name": name}}
    return {"name": "thread_name", "ph": "M", "pid": pid, "tid": tid, "args": {"name": name}}


class _Tids:
    def __init__(self) -> None:
        self.map = {}  # type: Dict[str, int]

    def get(self, label: str) -> int:
        if label not in self.map:
            self.map[label] = len(self.map) + 1
        return self.map[label]


def _expand_repeats(base_ts: float, repeats, max_periods: int):
    """Yield (ts, omitted_after) for the explicit periods of nested repeats."""

    positions = [base_ts]
    omitted = 0
    for label, count, stride in reversed(repeats):
        drawn = min(count, max_periods)
        omitted += max(count - drawn, 0)
        positions = [p + i * stride for p in positions for i in range(drawn)]
    return positions, omitted


def _group_events(
    group: RegionGroupResult, pid: int, tids: _Tids, events: List[Dict[str, Any]], max_periods: int,
) -> None:
    totals = {}  # type: Dict[str, float]
    for item in group.service_trace:
        repetitions = 1
        for _label, count, _stride in item.repeats:
            repetitions *= count
        totals[item.resource] = totals.get(item.resource, 0.0) + item.service_time_s * repetitions
        if item.offset_s is None:
            continue
        tid = tids.get("res:%s" % item.resource)
        positions, omitted = _expand_repeats(item.offset_s, item.repeats, max_periods)
        for ts in positions:
            events.append(_complete(
                item.op_id, pid, tid, ts, item.service_time_s,
                {"path": item.path, "kind": "resource_service"}, "service",
            ))
        if item.repeats:
            events.append(_instant(
                "%s periods" % item.op_id, pid, tid, item.offset_s,
                {
                    "repeats": [{"loop": l, "count": c, "stride_s": s} for l, c, s in item.repeats],
                    "drawn_periods": len(positions), "omitted_periods": omitted,
                    "note": "periods beyond drawn_periods are summarised, not drawn",
                },
            ))
    for item in group.completion_trace:
        if item.start_s is None:
            continue
        tid = tids.get("actor:%s" % item.actor)
        positions, omitted = _expand_repeats(item.start_s, item.repeats, max_periods)
        for ts in positions:
            events.append(_complete(
                item.op_id, pid, tid, ts, item.completion_latency_s,
                {"path": item.path, "kind": "completion"}, "completion",
            ))
    events.append(_instant(
        "summary", pid, tids.get("summary"), 0.0,
        {
            "total_s": group.total_s, "first_s": group.first_s, "steady_s": group.steady_s,
            "drain_s": group.drain_s, "strategy": group.strategy,
            "estimate_scope": group.estimate_scope,
            "cumulative_service_s": totals,
            "ii": None if group.ii is None else group.ii.to_dict(),
            "timed_bars": "only intervals with a known start are drawn",
        },
    ))


def perfetto_trace(result: AnalysisResult, *, max_periods: int = 8) -> Dict[str, Any]:
    """Return the Chrome Trace JSON object for ``result``."""

    events = []  # type: List[Dict[str, Any]]
    provenance = dict(result.diagnostics.provenance)
    context = provenance.get("context", {}) if isinstance(provenance.get("context"), dict) else {}
    label = "calibrated_model" if context.get("label", "default") != "default" else "theoretical_model"
    pid = 0
    launch_start = 0.0
    for name, launch in result.launches.items():
        pid += 1
        launch_start += result.program.gap_before(name)
        events.append(_meta(pid, None, "%s spatial" % name))
        timeline = dict(launch.provenance).get("unit_timeline", ())
        complete = bool(dict(launch.provenance).get("unit_timeline_complete", False))
        if complete and timeline:
            for index, slot, group_id, start, end in timeline:
                events.append(_complete(
                    "%s#%d" % (group_id, index), pid, slot + 1, launch_start + start, end - start,
                    {"work_unit": index, "group": group_id}, "spatial",
                ))
            for slot in range(len(launch.slot_completion_s)):
                events.append(_meta(pid, slot + 1, "slot %d" % slot))
        events.append(_instant(
            "launch summary", pid, 0, launch_start,
            {
                "kernel_body_s": launch.kernel_body_s, "total_s": launch.total_s,
                "waves": launch.waves, "slots": launch.slots, "work_units": launch.work_units,
                "spatial_scope": launch.spatial_scope,
                "unit_timeline_drawn": complete,
            },
        ))
        events.append(_meta(pid, 0, "launch"))
        region = result.regions.get(name)
        if region is not None:
            for group in region.groups:
                pid += 1
                events.append(_meta(pid, None, "%s work unit [%s]" % (name, group.work_group_id)))
                tids = _Tids()
                _group_events(group, pid, tids, events, max_periods)
                for label_name, tid in tids.map.items():
                    events.append(_meta(pid, tid, label_name))
        launch_start += launch.total_s
    return {
        "displayTimeUnit": "ns",
        "traceEvents": events,
        "metadata": {
            "tilesight_model": label,
            "model_version": result.model_version,
            "scenario": result.selected_scenario,
            "input_digest": result.input_digest,
            "units": "ts/dur in microseconds; args in seconds",
            "program_total_s": result.program.total_s,
            "work_unit_timelines": "relative to the work unit's own start (ts=0)",
        },
    }


def export_perfetto(result: AnalysisResult, path: str, *, max_periods: int = 8) -> str:
    with open(path, "w") as handle:
        json.dump(perfetto_trace(result, max_periods=max_periods), handle)
    return path


__all__ = ["export_perfetto", "perfetto_trace"]
