# TileLang frontend

Estimate how long a TileLang kernel runs on a GPU, and why, without running it.
The kernel is lowered once by TileLang's own pipeline; the program after
`PipelinePlanning` (tile operations, with the compiler's warp-specialized roles,
buffer stages and copy engines already decided) is translated into
`tilesight.modeling.program` and analyzed by TileSight. An optional Nsight Compute
(NCU) capture feeds measured clocks, traffic and residency back into the same
model. The analysis unit is one kernel with fixed shapes and configuration.

## Install

TileLang 0.1.12 or 0.1.14 (with TVM) is needed to profile a kernel; neither a GPU
nor a CUDA compiler is. TileLang is not a TileSight dependency, and
`import tilesight` never imports it.

```bash
pip install -e .
pip install tilelang==0.1.14
```

## Use

```bash
python -m tilesight.tilelang examples/tilelang/kernels/sum.py:build_kernel --target sm_90a \
  --factory-args '{"M": 8192, "N": 8192, "block_m": 2, "threads": 128, "dtype": "float16"}' \
  --output sum.json --report-dir sum-report
```

Exit status 0 means complete, 3 refused (the JSON names the reason), 2 failed.
Options: `--specialization` (keyword arguments of `get_tir`), `--pass-configs`
(default: the kernel's own), `--bindings` (shape symbols, integer scalar
parameters, or the contents of small inputs that steer control flow),
`--prim-func`, `--arch` and `--arch-profile` (default `H200_SXM` /
`set_to_microbench`), `--observation` (joint analysis).

```python
from tilesight.tilelang import profile_source_file

profile = profile_source_file("examples/tilelang/kernels/gemm.py", "build_kernel",
                              factory_args={"m": 8192, "n": 8192, "k": 8192}, target_arch="sm_90a")
print(profile.status, profile.estimate.kernel_body_s if profile.estimate else profile.refusal)
```

`profile_primfunc` takes a `PrimFunc` instead of a file. `examples/tilelang/run.py`
runs the example kernels in `examples/tilelang/cases.json` in three modes: `model`,
`capture` (NCU on a GPU) and `joint`.

`--report-dir` writes `report.json` (the source of truth), `report.html`,
`tables/*.csv` and `trace.perfetto.json`. Diagnoses: `pipeline_bottleneck`,
`compute_io_overlap`, `transfer_anomaly`, `cache_utilization` (DRAM re-reads per
tensor), `load_imbalance`, `latency_exposure` (few resident warps) and
`model_deviation` (with a measurement); thresholds are in `diagnose.POLICY`.

## Principles

1. **Model the compiled program.** The input is TileLang's module right after
   `PipelinePlanning`. Launch facts (threads, dynamic shared memory, instruction
   families) come from the device module of the same lowering.
2. **One lowering, public interfaces.** `tilelang.lower(..., enable_device_compile=False)`
   with a TVM pass instrument. A lowering that raises is a refusal
   (`LOWERING_FAILED`). No compiler runs.
3. **TileSight prices everything.** The frontend emits a TileSight program and
   never adds a cost of its own: throughput, cache behavior and pipelining are
   TileSight's, and the residency and the composition of resident CTAs use
   TileSight's own storage checks and resource bound (below).
4. **Refuse rather than default.** A missing fact or an unsupported construct
   produces a refusal code and message. Approximations that remain are written
   into the report notes.

## Capture (`capture.py`)

A pass instrument keeps the module after `PipelinePlanning`; exactly one such
snapshot and exactly one device kernel are required. The device module gives
threads, dynamic shared memory and the instruction families in the generated
source (TMA, cp.async, WGMMA, MMA). The compiler identity (version and a hash of
the installed sources and libraries) and the hash of the planned IR are recorded.

## Translation (`translate.py`, `emit.py`)

**Tensors and tiles.** A copy between a global tensor and an on-chip buffer
becomes a tile load or store. Per dimension the region is a tile walked by one
axis (a grid axis, a loop, or part of a grid axis split by `//` or `%`), a pick
(one index per CTA), or the whole dimension. A region that covers part of a
dimension reads its own declaration of that part: the same bytes, re-indexed.
Overlapping declarations of one tensor are allowed with a note, because the
cache model sees them as separate tensors.

**Roles and pipelines.** Branches on contiguous thread ranges around tile
operations are warp-specialized roles; a branch with only loads is a producer.
A shared buffer indexed by `(k + c) % S` in the loop is an `S`-slot ring; read
and write slot offsets and dataflow between operations give each operation's
iteration window. Hand-peeled first and drained last iterations are re-rolled.
Named barriers between roles become schedule constraints with the iteration
distance their generations imply.

**Element statements.** A store `t[idx] = f(...)` runs once per point of its
lane space: the enclosing `T.Parallel` loops, element loops written per thread
(vectorized, unrolled, or serial loops of stores with independent iterations),
and, for code outside `T.Parallel` that indexes by `threadIdx.x` or writes
thread-private storage, the threads themselves. A thread predicate such as
`tx % 32 == 31` selects an evenly spaced set of threads. Thread-private `local`
arrays are per-thread arrays; a `local.var` that holds data is one value per
lane or per thread; a `local.var` that holds an index is substituted. `f`'s work
is counted per element after common sub-expressions are merged and multiply-adds
fused. Reads and writes of global memory inside element statements become loads
and stores of the tile the lanes cover.

An accumulation `t = t + f`, `max`, `min`, or a select that keeps an extremum,
over lanes that do not index `t` becomes element work followed by a reduce. A
serial loop whose iterations depend on each other in any other way (a scan, a
recurrence) stays a loop, so TileSight sees its iterations in order.

**Trip counts and inputs.** Trip counts that depend on the CTA index are
evaluated per CTA and become TileSight work groups. `bindings` give shape
symbols, integer scalar parameters and the contents of small inputs that steer
control flow (sequence offsets, block tables).

## Residency and composition (`residency.py`, `backend.py`)

No compiler runs, so the residency is the largest number of CTAs per SM, at most
the architecture's `max_blocks_per_sm`, for which TileSight's own checks pass: the
program layer's capacity check of the declared shared and tensor memory
(`Launch.storage_bytes` against the capacity divided by the residency) and
TileSight's liveness analysis, which also holds the declared register fragments
against `register_capacity_per_sm`. Only declared storage counts: registers the
compiler allocates beyond the declared buffers, register buffers outside a
pipeline (the liveness analysis covers pipelines) and per-SM thread limits do
not, so the residency can exceed what the hardware allows. The payload names what
bounds it; the joint analysis adopts NCU's.

TileSight analyzes the program with one CTA per SM at full SM rates; with more
resident CTAs, TileSight's native per-SM resource bound (`resource_bound`) composes
them and `list_schedule` places the work units. A launch with fewer CTAs than one
wave holds is spread over the SMs first, as the native executor does. The program
layer's `equal_share` analysis at that residency is reported as an upper bound.

## Joint analysis (`joint.py`, `ncu.py`, `driver.py`)

`examples/tilelang/run.py --mode capture` (or `ncu.capture` with a request) runs the
kernel once under NCU through `python -m tilesight.tilelang.driver`, which compiles
the same request with TileLang and records the SHA-256 of the compiled CUDA
source. An observation is adopted only if its symbol, grid, block, dynamic shared
memory and SM count equal the model's and the compiled source hash matches.
Adopted facts are applied one at a time through TileSight's `Context`, so each
step's effect is reported: the SM clock, then L2/DRAM traffic (scaled per access
to the measured totals; not adopted when local-memory spills are present), then
NCU's resident-CTA limit when it differs and fits the declared storage. The
measured time is only compared, never fitted.
