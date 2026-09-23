"""Parametric FlashAttention-4 forward (Blackwell SM100) performance model.

model_fa4_us(b, h, kv_h, s, d, causal, arch) -> (total_us, breakdown dict)

Built on the overlap_analysis DAG framework.  The kernel configuration and
scheduler mapping are aligned to flash-attention commit eaf806d78fa9e18d
(`flash_attn/cute/flash_fwd_sm100.py`, `interface.py`, and
`tile_scheduler.py`).  The model captures:
  - the SM100 default 128x128 tile (causal => single-CTA, no 2CTA)
  - q_stage=2 ping-pong sub-tiles
  - packed-GQA grid construction (query heads folded into sequence)
  - TMEM channel (softmax t2r/r2t, correction O-rescale)
  - causal triangular KV-iteration count (per q-tile only attends [0, diag])
  - non-persistent wave tail, plus an optional blockIdx-order sensitivity for
    SingleTileLPTScheduler
  - per-tile prologue (pipe fill + Q load) and epilogue (O normalize + writeback)

The default return value is GPU kernel time, including one fixed 2 us launch
cost.  A separate 40 us host-dispatch constant remains available only for
reproducing the legacy wall-clock comparison.
"""
import heapq
import math
from tilesight.modeling._pipeline.overlap_analysis import (
    make_op_group, simulate_schedule,
)
from tilesight.modeling._pipeline.periodic_schedule import select_steady_ii

FLASH_ATTN_SOURCE_COMMIT = "eaf806d78fa9e18d9aadc323a77fbabe2c539331"
KERNEL_LAUNCH_OVERHEAD_US = 2.0
DISPATCH_OVERHEAD_US = 0.0
LEGACY_DISPATCH_OVERHEAD_US = 40.0

_PERIODIC_SEARCH_PROFILE_TRIALS = {
    'fast': 0,
    'reference': 20_000,
}


def _tile_cfg(d, causal):
    """Return the eaf806d SM100 default forward tile.

    The head-dimension-dependent 192x128 selector belongs to the SM90 branch in
    ``interface.py``.  SM100 leaves ``FwdConfig(128, 128, ...)`` unchanged for
    the dtypes/head dimensions modeled here.
    """
    if d > 192:
        raise NotImplementedError(
            "This model does not cover FA4's dedicated SM100 head_dim=256 kernel"
        )
    return 128, 128


def _l2_swizzle_size(seqlen_k, d, dv, dtype):
    """Mirror SingleTileLPTScheduler's power-of-two 50 MiB grouping rule."""
    size_one_kv_head = seqlen_k * (d + dv) * dtype
    scheduler_l2_budget = 50 * 1024 * 1024
    if size_one_kv_head > scheduler_l2_budget:
        return 1
    return 1 << int(math.log2(scheduler_l2_budget // size_one_kv_head))


def _lpt_grid_qblocks(num_q_blocks, num_head_batches, l2_swizzle):
    """Yield q-block indices in the static grid-x order used by FA4.

    Grid x is L2-swizzled into sections of head/batch coordinates.  Within each
    section, causal q-blocks are reversed so the longest blocks launch first.
    """
    section_start = 0
    while section_start < num_head_batches:
        section_size = min(l2_swizzle, num_head_batches - section_start)
        for q_block in range(num_q_blocks - 1, -1, -1):
            for _ in range(section_size):
                yield q_block
        section_start += section_size


def _grid_makespan(qblock_order, kv_iters, iter_latency, per_tile_overhead,
                   sm_count):
    """Approximate one-CTA-per-SM execution in ascending blockIdx order.

    CUDA does not guarantee CTA issue order.  This is therefore a sensitivity
    matching the kernel's logical blockIdx mapping, not a guaranteed schedule.
    """
    sm_heap = [(0.0, sm_id) for sm_id in range(sm_count)]
    heapq.heapify(sm_heap)
    sm_kv_iters = [0] * sm_count
    sm_tiles = [0] * sm_count

    for q_block in qblock_order:
        finish, sm_id = heapq.heappop(sm_heap)
        # ``per_tile_overhead`` already contains the serial first KV
        # iteration used to fill the pipeline.  Only the remaining iterations
        # advance at the steady-state II.
        steady_iters = max(kv_iters[q_block] - 1, 0)
        finish += steady_iters * iter_latency + per_tile_overhead
        sm_kv_iters[sm_id] += kv_iters[q_block]
        sm_tiles[sm_id] += 1
        heapq.heappush(sm_heap, (finish, sm_id))

    finish, sm_id = max(sm_heap)
    return finish, sm_kv_iters[sm_id], sm_tiles[sm_id]


def _aggregate_makespan(kv_iters, num_head_batches, iter_latency,
                        per_tile_overhead, sm_count):
    """Order-independent lower bound used for the primary wave estimate."""
    per_qblock = [
        max(x - 1, 0) * iter_latency + per_tile_overhead
        for x in kv_iters
    ]
    total_time = num_head_batches * sum(per_qblock)
    longest_tile = max(per_qblock)
    average_time = total_time / sm_count
    if longest_tile >= average_time:
        return longest_tile, max(kv_iters), 1.0
    return average_time, sum(kv_iters) * num_head_batches / sm_count, \
        len(kv_iters) * num_head_batches / sm_count


def _build_body(arch, M, N, D, Dv, q_stage, l2_hit, dtype, accum, max_util,
                correction_freq):
    """One KV-block loop body: load_K/V + per-stage QK/softmax/corr/PV."""
    g_load_k = make_op_group('load_K', arch,
        ddr_io=N * D * dtype * (1 - l2_hit), l2_io=N * D * dtype,
        smem_io=N * D * dtype, max_util=max_util)
    g_load_v = make_op_group('load_V', arch,
        ddr_io=N * Dv * dtype * (1 - l2_hit), l2_io=N * Dv * dtype,
        smem_io=N * Dv * dtype, max_util=max_util)
    groups = [g_load_k, g_load_v]
    for st in range(q_stage):
        g_qk = make_op_group(f'gemm_QK_s{st}', arch,
            tensor_flops=M * N * D * 2, max_util=max_util, depends_on=[g_load_k])
        g_sm = make_op_group(f'softmax_s{st}', arch,
            tmem_io=M * N * accum + M * N * dtype,
            cuda_flops=M * N * 6, sfu_flops=M * N + M,
            max_util=max_util, depends_on=[g_qk])
        g_corr = make_op_group(f'corr_s{st}', arch,
            tmem_io=int(2 * M * Dv * accum * correction_freq),
            cuda_flops=int(M * Dv * correction_freq),
            max_util=max_util, depends_on=[g_sm])
        g_pv = make_op_group(f'gemm_PV_s{st}', arch,
            tensor_flops=M * Dv * N * 2, max_util=max_util,
            depends_on=[g_sm, g_load_v, g_corr])
        groups += [g_qk, g_sm, g_corr, g_pv]
    return groups


def _epilogue_latency(arch, M, Dv, q_stage, dtype, accum, max_util):
    """O normalize (read TMEM, scale on cuda, write smem, TMA to gmem) + LSE."""
    g = make_op_group('epilogue', arch,
        tmem_io=q_stage * M * Dv * accum,           # read final O from TMEM
        cuda_flops=q_stage * M * Dv * 2,            # O *= 1/row_sum
        smem_io=q_stage * M * Dv * dtype,           # O -> smem
        ddr_io=q_stage * M * Dv * dtype,            # TMA smem -> gmem (O write, no L2 reuse)
        l2_io=q_stage * M * Dv * dtype,
        sfu_flops=q_stage * M,                      # LSE log2
        max_util=max_util)
    return g.latency


def model_fa4_us(b, h, s, d, *, kv_h=None, causal=True, arch,
                 dtype=2, accum=4, max_util=0.85, correction_freq=1.0,
                 dispatch_overhead_us=DISPATCH_OVERHEAD_US,
                 kernel_launch_overhead_us=KERNEL_LAUNCH_OVERHEAD_US,
                 scheduler_model='aggregate_lb',
                 inner_schedule_model='resource_ii',
                 periodic_search_profile='reference', verbose=False):
    """Estimate FA4 time with a selectable inner-II and search profile.

    ``periodic_search_profile='reference'`` preserves the 20k randomized
    proposal audit used by the periodic modes.  ``'fast'`` skips those random
    proposals while retaining the generic beam, source projection, and known
    constructive witnesses.  Resource-II mode never runs either search.
    """

    try:
        periodic_topology_trials = _PERIODIC_SEARCH_PROFILE_TRIALS[
            periodic_search_profile
        ]
    except (KeyError, TypeError):
        raise ValueError(
            "periodic_search_profile must be 'fast' or 'reference', got "
            f"{periodic_search_profile!r}"
        ) from None

    kv_h = kv_h if kv_h is not None else h
    if h % kv_h != 0:
        raise ValueError(f"h={h} must be divisible by kv_h={kv_h}")
    Dv = d
    tile_m, tile_n = _tile_cfg(d, causal)
    M, N = tile_m, tile_n
    qhead_per_kvhead = h // kv_h
    pack_gqa = qhead_per_kvhead > 1
    packed_seqlen = s * qhead_per_kvhead if pack_gqa else s
    scheduled_heads = kv_h if pack_gqa else h
    q_stage = 2 if packed_seqlen > tile_m else 1
    QM = q_stage * tile_m

    # --- grid / causal work ---
    nq = math.ceil(packed_seqlen / QM)
    num_head_batches = b * scheduled_heads
    grid = num_head_batches * nq

    def kv_iters(t):
        # BlockInfo divides the packed q coordinate by qhead_per_kvhead
        # before applying the causal boundary.
        q_end = min(math.ceil((t + 1) * QM / qhead_per_kvhead), s)
        if causal:
            return math.ceil(q_end / N)
        return math.ceil(s / N)

    kv_iters_by_qblock = [kv_iters(t) for t in range(nq)]
    iters_per_head_batch = sum(kv_iters_by_qblock)
    total_kv_iters = num_head_batches * iters_per_head_batch
    max_tile_iters = max(kv_iters_by_qblock)

    # The scheduler keeps a group of KV heads together in L2 while walking its
    # q-blocks.  The hit fraction is a simple repeated-q-block approximation.
    l2_hit = max(0.0, 1 - 1 / max(nq, 1))

    # --- per-KV-iter latency (pipelined, channel overlap) and serial (fill) ---
    body = _build_body(arch, M, N, d, Dv, q_stage, l2_hit, dtype, accum,
                       max_util, correction_freq)
    order = list(range(len(body)))
    resource_ii, _ = simulate_schedule(body, stage=2, order=order)
    iter_lat_serial, _ = simulate_schedule(body, stage=1, order=order)  # for prologue fill
    periodic_aux = {}

    def solve_periodic():
        # Import and search only if the common selector requests a constructive
        # endpoint.  The default resource-II DSE path never enters this factory.
        from .fa4_periodic import analyze_fa4_periodic

        envelope, kv_stage = analyze_fa4_periodic(
            body,
            tile_m=M,
            tile_n=N,
            d=d,
            dv=Dv,
            q_stage=q_stage,
            dtype_bytes=dtype,
            topology_trials=periodic_topology_trials,
        )
        periodic_aux['kv_stage'] = kv_stage
        periodic_aux['topology_trials'] = periodic_topology_trials
        return envelope

    ii_selection = select_steady_ii(
        resource_ii,
        inner_schedule_model,
        solve_periodic=solve_periodic,
    )
    iter_lat_pipe = ii_selection.ii
    periodic_envelope = ii_selection.envelope
    kv_stage = periodic_aux.get('kv_stage')
    if inner_schedule_model == 'resource_ii':
        ii_model = 'resource_ii_lower_bound'
    else:
        endpoint = (
            'best' if inner_schedule_model == 'periodic_best' else 'worst'
        )
        # Preserve the existing diagnostic spelling.  The common selector's
        # stronger scope vocabulary is exposed in separate fields below.
        legacy_scope = (
            'global' if periodic_envelope.search_complete else 'searched'
        )
        ii_model = f'periodic_{legacy_scope}_{endpoint}'

    # --- per-tile overhead: Q load + prologue fill (1 serial iter) + epilogue ---
    g_qload = make_op_group('load_Q', arch,
        ddr_io=QM * d * dtype, l2_io=QM * d * dtype, smem_io=QM * d * dtype,
        max_util=max_util)
    t_qload = g_qload.latency
    t_prologue = iter_lat_serial            # first KV iter cannot overlap (pipe fill)
    t_epilogue = _epilogue_latency(arch, M, Dv, q_stage, dtype, accum, max_util)
    per_tile_oh = t_qload + t_prologue + t_epilogue

    # CUDA does not promise the physical CTA issue order.  Use the
    # order-independent aggregate bound by default; the source-level blockIdx
    # mapping is available as an explicitly named sensitivity.
    sm = arch.sm_count
    l2_swizzle = _l2_swizzle_size(s, d, Dv, dtype)
    if scheduler_model == 'aggregate_lb':
        makespan, busy_kv_iters, tiles_busy = _aggregate_makespan(
            kv_iters_by_qblock, num_head_batches, iter_lat_pipe,
            per_tile_oh, sm,
        )
        scheduler = 'aggregate_order_independent_lower_bound'
    elif scheduler_model == 'sectioned_lpt':
        qblock_order = _lpt_grid_qblocks(nq, num_head_batches, l2_swizzle)
        makespan, busy_kv_iters, tiles_busy = _grid_makespan(
            qblock_order, kv_iters_by_qblock, iter_lat_pipe, per_tile_oh, sm
        )
        scheduler = 'sectioned_lpt_blockidx_order_sensitivity'
    else:
        raise ValueError(
            "scheduler_model must be 'aggregate_lb' or 'sectioned_lpt', got "
            f"{scheduler_model!r}"
        )
    # Every busy tile's first KV iteration is part of ``per_tile_oh``.  Keep
    # the actual KV count for diagnostics, but charge II only to subsequent
    # steady iterations.
    busy_steady_iters = max(busy_kv_iters - tiles_busy, 0.0)
    t_steady = busy_steady_iters * iter_lat_pipe
    t_tile_oh = tiles_busy * per_tile_oh
    kernel_body_us = makespan * 1e6
    gpu_us = kernel_body_us + kernel_launch_overhead_us
    total_us = gpu_us + dispatch_overhead_us

    # achieved TFLOPs (full causal-discounted flops over GPU time)
    flops = 2 * 2 * b * h * s * s * d * (0.5 if causal else 1.0)
    tflops = flops / (gpu_us * 1e-6) / 1e12 if gpu_us > 0 else 0

    bd = dict(tile_m=tile_m, tile_n=tile_n, q_stage=q_stage, QM=QM, nq=nq,
              pack_gqa=pack_gqa, qhead_per_kvhead=qhead_per_kvhead,
              scheduled_heads=scheduled_heads,
              grid=grid, total_kv_iters=total_kv_iters, max_tile_iters=max_tile_iters,
              waves=grid / sm, tiles_busy=tiles_busy, busy_kv_iters=busy_kv_iters,
              busy_steady_iters=busy_steady_iters,
              scheduler=scheduler, scheduler_model=scheduler_model,
              ii_model=ii_model, inner_schedule_model=inner_schedule_model,
              periodic_search_profile=periodic_search_profile,
              periodic_topology_trials=periodic_aux.get('topology_trials'),
              ii_selection_label=ii_selection.label,
              ii_selection_scope=ii_selection.endpoint_scope,
              ii_is_constructive=ii_selection.is_constructive,
              ii_search_complete=ii_selection.search_complete,
              ii_selected_ns=ii_selection.ii * 1e9,
              l2_swizzle=l2_swizzle,
              l2_hit=l2_hit, iter_lat_pipe_ns=iter_lat_pipe * 1e9,
              resource_ii_lb_ns=resource_ii * 1e9,
              iter_lat_serial_ns=iter_lat_serial * 1e9, per_tile_oh_ns=per_tile_oh * 1e9,
              t_steady_us=t_steady * 1e6, t_tile_oh_us=t_tile_oh * 1e6,
              kernel_body_us=kernel_body_us,
              kernel_launch_us=kernel_launch_overhead_us,
              gpu_us=gpu_us, dispatch_us=dispatch_overhead_us, total_us=total_us,
              tflops=tflops)
    if periodic_envelope is not None:
        bd.update(
            kv_stage=kv_stage,
            periodic_best_ii_ns=periodic_envelope.best.ii * 1e9,
            periodic_worst_ii_ns=periodic_envelope.worst.ii * 1e9,
            periodic_spread_ns=periodic_envelope.overlap_sensitivity * 1e9,
            periodic_spread_pct=(
                periodic_envelope.overlap_sensitivity
                / periodic_envelope.best.ii
                * 100
            ),
            periodic_orderings_explored=periodic_envelope.orderings_explored,
            periodic_infeasible_orderings=periodic_envelope.infeasible_orderings,
            periodic_unsupported_orderings=(
                periodic_envelope.unsupported_orderings
            ),
            periodic_search_complete=periodic_envelope.search_complete,
            periodic_endpoint_scope=(
                'global' if periodic_envelope.search_complete else 'searched'
            ),
            periodic_best_resource_orders=dict(
                periodic_envelope.best.resource_orders
            ),
            periodic_worst_resource_orders=dict(
                periodic_envelope.worst.resource_orders
            ),
        )
    if verbose:
        for k, v in bd.items():
            print(f"    {k:>20} = {v:.3f}" if isinstance(v, float) else f"    {k:>20} = {v}")
    return total_us, bd
