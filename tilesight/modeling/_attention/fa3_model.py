"""Source-scoped FlashAttention-3 forward model for Hopper (SM90).

The structural constants follow the local flash-attention checkout at
``2e53092aa70fccd3f04013a01a52dc20c619e62b``:

* ``hopper/tile_size.h`` selects the BF16 tile;
* ``hopper/flash_fwd_launch_template.h`` fixes the K/V pipeline at two stages
  and selects the persistent scheduler on SM90;
* ``hopper/block.h`` supplies the causal and PackGQA KV bounds; and
* ``hopper/tile_scheduler.hpp`` supplies the static or sectioned-LPT order.

The default is deliberately the resource-II lower bound.  The shared
``inner_schedule_model`` interface can opt into constructive periodic-DAG best
or worst endpoints; ``steady_ii_scheduler`` remains only as a compatibility
hook.  Neither path may lower the resource-II bound.  The fixed 2 us GPU launch
cost is added exactly once after the modeled kernel body.  Host/Python dispatch
is a separate optional term because it is not an intrinsic kernel cost.

Scope: fixed-length, self-attention, BF16, d in {64, 128}, no split-K/local/
paged-KV/dropout.  Unsupported configurations fail loudly instead of silently
using a nearby kernel configuration.
"""

from __future__ import annotations

import heapq
import math
from typing import Callable, Iterable, Optional

from tilesight.modeling._pipeline.overlap_analysis import (
    OpGroup,
    make_op_group,
    simulate_schedule,
)
from tilesight.modeling._pipeline.periodic_schedule import (
    IIScheduleSelection,
    select_steady_ii,
)


AUDITED_FLASH_ATTN_SOURCE_COMMIT = "2e53092aa70fccd3f04013a01a52dc20c619e62b"
KERNEL_LAUNCH_OVERHEAD_US = 2.0
HOPPER_KV_PIPELINE_STAGES = 2
HOPPER_L2_SCHEDULER_BUDGET_BYTES = 32 * 1024 * 1024


def _ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def _tile_config(d: int, causal: bool) -> tuple[int, int, bool, bool]:
    """Return (block_m, block_n, mma_pv_is_rs, intra_wg_overlap)."""
    if d == 64:
        # tile_size_fwd_sm90: causal selects N=128 and RS PV; non-causal
        # selects the 192x192 SS-PV variant.
        return (192, 128, True, True) if causal else (192, 192, False, True)
    if d == 128:
        return (128, 128, True, True) if causal else (128, 176, True, True)
    raise NotImplementedError(
        "FA3 Hopper model is source-validated only for BF16 d=64 or d=128; "
        f"got d={d}"
    )


def _should_pack_gqa(s: int, qheads_per_kvhead: int, block_m: int) -> bool:
    """Mirror ``heuristics.h::should_pack_gqa`` for fixed-length inputs."""
    if qheads_per_kvhead == 1:
        return False
    unpacked_eff = s / (_ceil_div(s, block_m) * block_m)
    packed_s = s * qheads_per_kvhead
    packed_eff = packed_s / (_ceil_div(packed_s, block_m) * block_m)
    return unpacked_eff < 0.9 * packed_eff


def _kv_iters_by_qblock(
    s: int,
    block_m: int,
    block_n: int,
    qheads_per_kvhead: int,
    pack_gqa: bool,
    causal: bool,
) -> list[int]:
    """Mirror the equal-Q/K-length part of ``BlockMN::get_n_block_min_max``."""
    packed_s = s * qheads_per_kvhead if pack_gqa else s
    num_qblocks = _ceil_div(packed_s, block_m)
    full_kv_iters = _ceil_div(s, block_n)
    if not causal:
        return [full_kv_iters] * num_qblocks

    out = []
    for m_block in range(num_qblocks):
        m_idx_max = (m_block + 1) * block_m
        if pack_gqa:
            # C++: divide(m_idx_max - 1) + 1, i.e. ceil(m_idx_max / ratio).
            m_idx_max = _ceil_div(m_idx_max, qheads_per_kvhead)
        out.append(min(full_kv_iters, _ceil_div(m_idx_max, block_n)))
    return out


def _l2_swizzle(
    s: int,
    d: int,
    dtype_bytes: int,
    qheads_per_kvhead: int,
    pack_gqa: bool,
) -> int:
    """Mirror DynamicPersistentTileScheduler's 32 MiB power-of-two section."""
    one_kv_head = s * (d + d) * dtype_bytes
    if one_kv_head > HOPPER_L2_SCHEDULER_BUDGET_BYTES:
        base = 1
    else:
        base = 1 << int(math.log2(HOPPER_L2_SCHEDULER_BUDGET_BYTES // one_kv_head))
    return base * (1 if pack_gqa else qheads_per_kvhead)


def _average_kv_miss_fraction(
    kv_iters: Iterable[int],
    full_kv_iters: int,
    qheads_per_kvhead: int,
    pack_gqa: bool,
    one_kv_head_bytes: int,
) -> float:
    """Cold-L2 compulsory K/V fraction under source scheduler locality.

    The longest tile is visited first, so one full K/V head is the compulsory
    traffic and later q-blocks (and unpacked GQA query heads) reuse it.  When a
    K/V head exceeds the source scheduler's L2 budget, reuse is not assumed.
    """
    if one_kv_head_bytes > HOPPER_L2_SCHEDULER_BUDGET_BYTES:
        return 1.0
    reuse_copies = 1 if pack_gqa else qheads_per_kvhead
    total_block_loads = sum(kv_iters) * reuse_copies
    if total_block_loads <= 0:
        return 1.0
    return min(1.0, full_kv_iters / total_block_loads)


def _build_iteration_body(
    arch,
    block_m: int,
    block_n: int,
    d: int,
    dtype_bytes: int,
    accum_bytes: int,
    kv_miss_fraction: float,
    mma_pv_is_rs: bool,
    max_util: float,
) -> list[OpGroup]:
    """Build one source-ordered K/V iteration of the Hopper mainloop."""
    kv_bytes = block_n * d * dtype_bytes
    load_k = make_op_group(
        "load_K",
        arch,
        ddr_io=kv_bytes * kv_miss_fraction,
        l2_io=kv_bytes,
        smem_io=kv_bytes,
        max_util=max_util,
    )
    load_v = make_op_group(
        "load_V",
        arch,
        ddr_io=kv_bytes * kv_miss_fraction,
        l2_io=kv_bytes,
        smem_io=kv_bytes,
        max_util=max_util,
    )
    qk = make_op_group(
        "gemm_QK",
        arch,
        tensor_flops=2 * block_m * block_n * d,
        max_util=max_util,
        depends_on=[load_k],
    )
    # The operation counts are analytical (not fitted): max/sum/scale/convert
    # plus exp2 per score.  SS-PV writes and rereads P through shared memory;
    # RS-PV consumes P from registers.
    p_smem_io = 0 if mma_pv_is_rs else 2 * block_m * block_n * dtype_bytes
    softmax = make_op_group(
        "online_softmax",
        arch,
        smem_io=p_smem_io,
        cuda_flops=6 * block_m * block_n,
        sfu_flops=block_m * block_n + block_m,
        max_util=max_util,
        depends_on=[qk],
    )
    rescale_o = make_op_group(
        "rescale_O",
        arch,
        cuda_flops=block_m * d,
        max_util=max_util,
        depends_on=[softmax],
    )
    pv = make_op_group(
        "gemm_PV",
        arch,
        tensor_flops=2 * block_m * block_n * d,
        max_util=max_util,
        depends_on=[load_v, softmax, rescale_o],
    )
    return [load_k, load_v, qk, softmax, rescale_o, pv]


def _steady_ii(
    body: list[OpGroup],
    inner_schedule_model: str,
    steady_ii_scheduler: Optional[Callable],
) -> tuple[float, float, str, dict, Optional[IIScheduleSelection]]:
    """Return selected II plus the common selection semantics.

    ``steady_ii_scheduler`` is retained as a compatibility hook.  The stable
    path is ``inner_schedule_model``; its periodic factory is deliberately
    lazy so the default DSE path neither imports nor runs the DAG search.
    """
    order = list(range(len(body)))
    resource_ii, utilization = simulate_schedule(
        body, stage=HOPPER_KV_PIPELINE_STAGES, order=order
    )
    if steady_ii_scheduler is None:
        def solve_periodic():
            from .fa3_periodic import analyze_fa3_periodic

            return analyze_fa3_periodic(
                tuple(body), pipeline_stages=HOPPER_KV_PIPELINE_STAGES
            )

        selection = select_steady_ii(
            resource_ii,
            inner_schedule_model,
            solve_periodic=solve_periodic,
        )
        envelope = selection.envelope
        metadata = dict(utilization)
        if envelope is not None:
            metadata.update(
                {
                    "periodic_selection": (
                        "best"
                        if inner_schedule_model == "periodic_best"
                        else "worst"
                    ),
                    "periodic_best_ii_ns": envelope.best.ii * 1e9,
                    "periodic_worst_ii_ns": envelope.worst.ii * 1e9,
                    "periodic_overlap_spread_ns": (
                        envelope.overlap_sensitivity * 1e9
                    ),
                    "periodic_resource_lb_ns": (
                        envelope.lower_bounds.resource_ii * 1e9
                    ),
                    "periodic_recurrence_lb_ns": (
                        envelope.lower_bounds.recurrence_ii * 1e9
                    ),
                    "periodic_credit_lb_ns": (
                        envelope.lower_bounds.credit_ii * 1e9
                    ),
                    "periodic_orderings_explored": envelope.orderings_explored,
                    "periodic_infeasible_orderings": (
                        envelope.infeasible_orderings
                    ),
                    "periodic_unsupported_orderings": (
                        envelope.unsupported_orderings
                    ),
                    "periodic_search_complete": envelope.search_complete,
                    "periodic_best_phase_starts_ns": {
                        name: value * 1e9
                        for name, value in envelope.best.phase_starts.items()
                    },
                    "periodic_worst_phase_starts_ns": {
                        name: value * 1e9
                        for name, value in envelope.worst.phase_starts.items()
                    },
                    "periodic_best_resource_orders": dict(
                        envelope.best.resource_orders
                    ),
                    "periodic_worst_resource_orders": dict(
                        envelope.worst.resource_orders
                    ),
                }
            )
        # Preserve the pre-selector diagnostic strings while exposing the
        # stable selector label separately in the public breakdown.
        if inner_schedule_model == "resource_ii":
            ii_model = "resource_ii"
        else:
            endpoint = (
                "fa3_periodic_best"
                if inner_schedule_model == "periodic_best"
                else "fa3_periodic_worst"
            )
            ii_model = f"periodic:{endpoint}"
        return selection.ii, resource_ii, ii_model, metadata, selection

    if inner_schedule_model != "resource_ii":
        raise ValueError(
            "steady_ii_scheduler is a legacy compatibility hook and cannot "
            "be combined with a non-default inner_schedule_model"
        )

    result = steady_ii_scheduler(
        groups=tuple(body),
        pipeline_stages=HOPPER_KV_PIPELINE_STAGES,
        resource_ii_s=resource_ii,
    )
    metadata = {}
    if isinstance(result, tuple):
        selected_ii, metadata = result
    else:
        selected_ii = result
    selected_ii = float(selected_ii)
    if not math.isfinite(selected_ii) or selected_ii <= 0:
        raise ValueError(f"periodic scheduler returned invalid II {selected_ii!r}")
    if selected_ii + 1e-18 < resource_ii:
        raise ValueError(
            "periodic scheduler returned an II below the resource lower bound: "
            f"{selected_ii:.6e} < {resource_ii:.6e}"
        )
    name = getattr(steady_ii_scheduler, "__name__", type(steady_ii_scheduler).__name__)
    return (
        selected_ii,
        resource_ii,
        f"periodic:{name}",
        {**utilization, **metadata},
        None,
    )


def _sectioned_lpt_qblocks(
    num_qblocks: int, num_head_batches: int, swizzle: int
) -> Iterable[int]:
    section_start = 0
    while section_start < num_head_batches:
        section_size = min(swizzle, num_head_batches - section_start)
        for qblock in range(num_qblocks - 1, -1, -1):
            for _ in range(section_size):
                yield qblock
        section_start += section_size


def _source_makespan(
    tile_times: list[float],
    kv_iters: list[int],
    num_head_batches: int,
    sm_count: int,
    causal: bool,
    swizzle: int,
) -> tuple[float, int, int, str]:
    grid = len(tile_times) * num_head_batches
    if not causal:
        # StaticPersistentTileScheduler advances by gridDim.x == num_sm.
        critical_tiles = _ceil_div(grid, sm_count)
        return (
            critical_tiles * tile_times[0],
            critical_tiles,
            critical_tiles * kv_iters[0],
            "static_persistent_stride",
        )

    heap = [(0.0, sm) for sm in range(sm_count)]
    heapq.heapify(heap)
    sm_tiles = [0] * sm_count
    sm_iters = [0] * sm_count
    order = _sectioned_lpt_qblocks(len(tile_times), num_head_batches, swizzle)
    for qblock in order:
        finish, sm = heapq.heappop(heap)
        finish += tile_times[qblock]
        sm_tiles[sm] += 1
        sm_iters[sm] += kv_iters[qblock]
        heapq.heappush(heap, (finish, sm))
    makespan, critical_sm = max(heap)
    return (
        makespan,
        sm_tiles[critical_sm],
        sm_iters[critical_sm],
        "dynamic_persistent_sectioned_lpt",
    )


def _aggregate_lower_bound(
    tile_times: list[float], kv_iters: list[int], num_head_batches: int, sm_count: int
) -> tuple[float, float, float, str]:
    total = num_head_batches * sum(tile_times)
    longest_idx = max(range(len(tile_times)), key=tile_times.__getitem__)
    average = total / sm_count
    if tile_times[longest_idx] >= average:
        return tile_times[longest_idx], 1.0, float(kv_iters[longest_idx]), "aggregate_lb"
    scale = num_head_batches / sm_count
    return average, len(tile_times) * scale, sum(kv_iters) * scale, "aggregate_lb"


def model_fa3_us(
    b: int,
    h: int,
    s: int,
    d: int,
    *,
    arch,
    kv_h: Optional[int] = None,
    causal: bool = False,
    dtype_bytes: int = 2,
    accum_bytes: int = 4,
    max_util: float = 0.85,
    host_dispatch_overhead_us: float = 0.0,
    scheduler_model: str = "source",
    inner_schedule_model: str = "resource_ii",
    steady_ii_scheduler: Optional[Callable] = None,
    verbose: bool = False,
) -> tuple[float, dict]:
    """Model one fixed-length BF16 FlashAttention-3 forward launch.

    Returns ``(total_us, breakdown)``.  ``total_us`` contains the modeled GPU
    body, one fixed 2 us launch cost, and the explicitly supplied host dispatch
    term.  The latter defaults to zero and is never folded into the 2 us term.
    """
    if min(b, h, s, d) <= 0:
        raise ValueError("b, h, s and d must all be positive")
    if dtype_bytes != 2:
        raise NotImplementedError("this source-scoped FA3 model covers BF16/FP16 only")
    if accum_bytes != 4:
        raise NotImplementedError("the audited FA3 path uses FP32 accumulators")
    if max_util <= 0 or max_util > 1:
        raise ValueError(f"max_util must be in (0, 1], got {max_util}")
    if host_dispatch_overhead_us < 0:
        raise ValueError("host_dispatch_overhead_us cannot be negative")
    kv_h = h if kv_h is None else kv_h
    if kv_h <= 0 or h % kv_h:
        raise ValueError(f"h={h} must be divisible by kv_h={kv_h}")

    block_m, block_n, mma_pv_is_rs, intra_wg_overlap = _tile_config(d, causal)
    qheads_per_kvhead = h // kv_h
    pack_gqa = _should_pack_gqa(s, qheads_per_kvhead, block_m)
    scheduled_heads = kv_h if pack_gqa else h
    num_head_batches = b * scheduled_heads
    kv_iters = _kv_iters_by_qblock(
        s, block_m, block_n, qheads_per_kvhead, pack_gqa, causal
    )
    num_qblocks = len(kv_iters)
    grid_work_tiles = num_head_batches * num_qblocks
    full_kv_iters = _ceil_div(s, block_n)
    one_kv_head_bytes = s * (d + d) * dtype_bytes
    kv_miss_fraction = _average_kv_miss_fraction(
        kv_iters,
        full_kv_iters,
        qheads_per_kvhead,
        pack_gqa,
        one_kv_head_bytes,
    )

    body = _build_iteration_body(
        arch,
        block_m,
        block_n,
        d,
        dtype_bytes,
        accum_bytes,
        kv_miss_fraction,
        mma_pv_is_rs,
        max_util,
    )
    steady_ii_s, resource_ii_s, ii_model, ii_metadata, ii_selection = _steady_ii(
        body, inner_schedule_model, steady_ii_scheduler
    )
    serial_body_s = sum(group.latency for group in body)

    q_bytes = block_m * d * dtype_bytes
    q_load_s = make_op_group(
        "load_Q",
        arch,
        ddr_io=q_bytes,
        l2_io=q_bytes,
        smem_io=q_bytes,
        max_util=max_util,
    ).latency
    # PackGQA disables TMA-O, so it does not stage O through shared memory.
    o_bytes = block_m * d * dtype_bytes
    epilogue_s = make_op_group(
        "normalize_store_O",
        arch,
        ddr_io=o_bytes,
        l2_io=o_bytes,
        smem_io=0 if pack_gqa else o_bytes,
        cuda_flops=2 * block_m * d,
        sfu_flops=block_m,
        max_util=max_util,
    ).latency

    prologue_per_tile_s = q_load_s + serial_body_s
    tile_times = [
        prologue_per_tile_s + max(n_iters - 1, 0) * steady_ii_s + epilogue_s
        for n_iters in kv_iters
    ]
    swizzle = _l2_swizzle(
        s, d, dtype_bytes, qheads_per_kvhead, pack_gqa
    )
    if scheduler_model == "source":
        makespan_s, critical_tiles, critical_iters, scheduler = _source_makespan(
            tile_times,
            kv_iters,
            num_head_batches,
            arch.sm_count,
            causal,
            swizzle,
        )
    elif scheduler_model == "aggregate_lb":
        makespan_s, critical_tiles, critical_iters, scheduler = _aggregate_lower_bound(
            tile_times, kv_iters, num_head_batches, arch.sm_count
        )
    else:
        raise ValueError("scheduler_model must be 'source' or 'aggregate_lb'")

    gpu_body_us = makespan_s * 1e6
    kernel_us = gpu_body_us + KERNEL_LAUNCH_OVERHEAD_US
    total_us = kernel_us + host_dispatch_overhead_us
    useful_flops = 4 * b * h * d * (
        s * (s + 1) / 2 if causal else s * s
    )
    tflops = useful_flops / (kernel_us * 1e-6) / 1e12 if kernel_us else 0.0

    critical_tiles_f = float(critical_tiles)
    critical_iters_f = float(critical_iters)
    breakdown = {
        "tile_m": block_m,
        "tile_n": block_n,
        "mma_pv_is_rs": mma_pv_is_rs,
        "intra_wg_overlap": intra_wg_overlap,
        "kv_pipeline_stages": HOPPER_KV_PIPELINE_STAGES,
        "pack_gqa": pack_gqa,
        "qheads_per_kvhead": qheads_per_kvhead,
        "scheduled_heads": scheduled_heads,
        "nq": num_qblocks,
        "grid_work_tiles": grid_work_tiles,
        "launched_persistent_ctas": arch.sm_count,
        "waves": grid_work_tiles / arch.sm_count,
        "kv_iters_by_qblock": tuple(kv_iters),
        "total_kv_iters": num_head_batches * sum(kv_iters),
        "critical_tiles": critical_tiles,
        "critical_kv_iters": critical_iters,
        "l2_swizzle": swizzle,
        "kv_l2_hit_fraction": 1.0 - kv_miss_fraction,
        "scheduler": scheduler,
        "scheduler_model": scheduler_model,
        "inner_schedule_model": inner_schedule_model,
        "ii_model": ii_model,
        "ii_selection_label": (
            ii_model if ii_selection is None else ii_selection.label
        ),
        "ii_selection_scope": (
            "legacy_hook" if ii_selection is None else ii_selection.endpoint_scope
        ),
        "ii_is_constructive": (
            None if ii_selection is None else ii_selection.is_constructive
        ),
        "ii_search_complete": (
            None if ii_selection is None else ii_selection.search_complete
        ),
        "ii_selected_ns": steady_ii_s * 1e9,
        "steady_ii_ns": steady_ii_s * 1e9,
        "resource_ii_lb_ns": resource_ii_s * 1e9,
        "serial_body_ns": serial_body_s * 1e9,
        "q_load_ns": q_load_s * 1e9,
        "epilogue_ns": epilogue_s * 1e9,
        "t_prologue_us": critical_tiles_f * prologue_per_tile_s * 1e6,
        "t_steady_us": max(critical_iters_f - critical_tiles_f, 0.0)
        * steady_ii_s
        * 1e6,
        "t_epilogue_us": critical_tiles_f * epilogue_s * 1e6,
        "gpu_body_us": gpu_body_us,
        "kernel_launch_us": KERNEL_LAUNCH_OVERHEAD_US,
        "kernel_us": kernel_us,
        "host_dispatch_us": host_dispatch_overhead_us,
        "total_us": total_us,
        "tflops": tflops,
        "ii_metadata": ii_metadata,
        "source_commit": AUDITED_FLASH_ATTN_SOURCE_COMMIT,
    }
    if verbose:
        for key, value in breakdown.items():
            print(f"    {key:>24} = {value}")
    return total_us, breakdown
