# GPU measurements

Use Python 3.10+ and a compatible CUDA or ROCm toolkit on the target machine.
Run commands from the repository root on idle, full GPUs. `--quick` runs a short
smoke check. The runners leave clocks, power limits and architecture profiles unchanged.

## NVIDIA

```bash
python micro_benchmark/run.py --gpu auto --device 0 --quick
python micro_benchmark/run.py --gpu b200 --device 0
python micro_benchmark/run.py --gpu auto --device 0 --bench dram --dram-blocks 1,2,4,8,16,32,64,128,256,512,1024 --data-seed 1729
python micro_benchmark/run.py --gpu b200 --build-only
```

`--device` is an ordinal inside `CUDA_VISIBLE_DEVICES`. The runner pins child
processes to that GPU's UUID. Use `NVCC=/path/to/nvcc` or `--nvcc` to select CUDA.
Explicit profiles are `h200`, `b200` and `b6000` (RTX PRO 6000 Blackwell).
`auto` supports compiler-compatible NVIDIA GPUs from SM75, including H100.

| GPU | Generic target | Specialized probes |
| --- | --- | --- |
| H100/H200 | SM90 | WGMMA, SM90a |
| B200 | SM100 | TCGEN05/TMEM, SM100a |
| RTX PRO 6000 Blackwell | SM120 | Generic probes and cuBLASLt |

The compiler must support the detected target. Specialized instructions require
the exact SM above. MIG is unsupported. Use `--list` for probes and `--dry-run`
for an offline plan with an explicit profile or `--arch`.

Data policy 2 uses reproducible nonzero operands. `--data-seed` controls hashed
DRAM/L2/shared/TMEM payloads and is recorded with each configuration. WGMMA and
TCGEN05 use nonzero input. Compare runs with matching data policies and settings.
DRAM working sets are at least 8× L2. Small grids increase iterations to cover
the allocation, while larger grids can issue more total traffic. Copy counts
both read and write bytes. The default sweep uses 1×/2×/4× the SM count.

## AMD HIP/ROCm

```bash
python micro_benchmark/run_amd.py --device 0 --quick
HIPCC=/path/to/hipcc python micro_benchmark/run_amd.py --device 0 --bench dram --dram-blocks 1,8,32,128,512
python micro_benchmark/run_amd.py --device 0 --bench rocblas_gemm
python micro_benchmark/run_amd.py --arch gfx942 --dry-run
python micro_benchmark/run_amd.py --arch gfx950 --build-only
```

Device detection automatically supplies the full `gcnArchName` to `hipcc`.
`--device` respects inherited `HIP_VISIBLE_DEVICES` and `ROCR_VISIBLE_DEVICES`.
A logical device can represent one GCD or partition. `HIPCC` or `--hipcc` selects
the compiler. Core probes cover DRAM copy, warm-cache reads, FP32/FP64, sqrt and
launch timing with deterministic nonzero inputs. Optional rocBLAS GEMM uses
FP16 inputs, FP32 accumulation/output and sampled reference checks.
AMD probes have not yet been compiled with hipcc or validated on AMD hardware.

## NVIDIA multi-GPU

```bash
python micro_benchmark/multi_gpu.py --devices 0,1 --backend p2p --quick
python micro_benchmark/multi_gpu.py --devices 0,1 --backend all --dry-run
```

Device order follows the inherited CUDA mask. P2P checks every transferred byte
for each directed pair. Copy time in µs includes source-stream submission and queue
overhead. It measures the complete transfer path. Unsupported pairs have no measurement.

Install [nccl-tests](https://github.com/NVIDIA/nccl-tests) separately:

```bash
python micro_benchmark/multi_gpu.py --devices 0,1 --backend nccl --quick \
  --nccl-all-reduce /path/to/nccl-tests/build/all_reduce_perf
```

AllReduce uses one rank per selected GPU and checks correctness. Unset
`NCCL_TESTS_SPLIT` and `NCCL_TESTS_SPLIT_MASK`. Time, algorithm bandwidth and bus
bandwidth remain separate for in-place and out-of-place results.

NVSHMEM requires two GPUs, installed **device/pt-to-pt** benchmarks and Hydra:

```bash
python micro_benchmark/multi_gpu.py --devices 0,1 --backend nvshmem --quick \
  --nvshmem-launcher /path/to/hydra/bin/nvshmrun \
  --nvshmem-put-latency /path/to/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_latency \
  --nvshmem-put-bw /path/to/nvshmem/bin/perftest/device/pt-to-pt/shmem_put_bw
```

Configure library paths and [NVSHMEM bootstrap](https://docs.nvidia.com/nvshmem/api/latest/using.html#running-nvshmem-programs).
The adapter uses two local Hydra PEs and defaults to PMI when unset. Installed
CLI/output formats are checked. Native put latency (µs) and bandwidth (GB/sec) retain their
scope and synchronization overhead. Correctness is not independently verified.
Bandwidth starts at 8 bytes per CTA, with 32 CTAs by default. `--nvshmem-ctas 1`
allows 8-byte payloads. These optional adapters need target-machine validation.

## Units and local output

Bandwidth uses GB/s, with GiB/s also recorded for DRAM. Compute uses TFLOP/s and SFU uses TOP/s.
Shared-memory/TMEM rates are bytes/cycle/SM. TMEM reads wait before register reuse.
Pointer-chase latency includes loop
overhead. AMD warm-cache rates cover the cache hierarchy and its launch timing
includes enqueue/completion. NCCL bus bandwidth is an adjusted collective metric.

Use each runner's `--help` for size, iteration, compiler and timeout options.
Outputs go to ignored `micro_benchmark/results/` or a new `--output` directory.
JSON, CSV and raw logs retain identities, commands, hashes, units and status.
Missing dependencies and unsupported probes are explicit. Failures return nonzero.
Multi-GPU `--backend all` can return partial success when some backends measured
and others were unavailable. It returns nonzero if none measured or any failed.
