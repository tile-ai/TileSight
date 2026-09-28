# TileSight

TileSight models GPU kernels from tile shapes, data movement, software pipelines
and hardware profiles. It estimates operation costs, cache traffic and execution
time on the CPU. CUDA and HIP probes measure the target GPU separately.

## Install

Python 3.10+ is required. Modeling examples run on the CPU.

```bash
git clone https://github.com/tile-ai/TileSight.git
cd TileSight
python -m pip install -e .
```

## Model a kernel

```python
import tilesight as sight
from tilesight.arch.h200_sxm import H200_SXM
from tilesight.modeling.program import examples as ex

program = ex.gemm_program(m=1024, n=1024, k=1024, dtype="bf16")
result = sight.analyze(
    program, H200_SXM().set_to_microbench(),
    options=sight.Options(cache="fast", ii_mode="periodic_best"),
)
print("Predicted kernel body (s):", result.launches["main"].kernel_body_s)
```

Build your own programs with [KernelBuilder](tilesight/modeling/program/builder.py).
The [example builders](tilesight/modeling/program/examples/) and
[result types](tilesight/modeling/program/results.py) document shapes, schedules,
traffic and timing. Profiles contain measured, derived or inherited parameters.

```bash
python examples/periodic_schedule_basics.py
python examples/fa3_fa4_schedule_modes.py
python examples/flashmla_decode.py --num-splits 4
```

## Model a TileLang kernel

The [TileLang frontend](tilesight/tilelang/README.md) lowers a TileLang kernel once
and translates the compiled schedule into a program; it needs TileLang 0.1.12 or
0.1.14, no GPU.

```bash
python -m pip install tilelang==0.1.14
python -m tilesight.tilelang examples/tilelang/kernels/gemm.py:build_kernel --target sm_90a \
  --factory-args '{"m": 8192, "n": 8192, "k": 8192}' --output gemm.json --report-dir gemm-report
```

The [GPU measurement guide](micro_benchmark/README.md) covers NVIDIA SM and AMD
gfx selection, DRAM concurrency scans and optional communication tests.

## Layout and tests

| Path | Purpose |
| --- | --- |
| `tilesight/arch/` | GPU architecture profiles |
| `tilesight/modeling/` | Program frontend, cost/cache models and scheduling |
| `tilesight/tilelang/` | TileLang frontend: compiled TileLang kernels to programs |
| `tilesight/util/` | Shared utilities |
| `examples/` | Runnable modeling examples |
| `micro_benchmark/` | CUDA/HIP probes and communication adapters |

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Generated measurements and binaries remain local. [MIT license](LICENSE).
