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

The [GPU measurement guide](micro_benchmark/README.md) covers NVIDIA SM and AMD
gfx selection, DRAM concurrency scans and optional communication tests.

## Layout and tests

| Path | Purpose |
| --- | --- |
| `tilesight/arch/` | GPU architecture profiles |
| `tilesight/modeling/` | Program frontend, cost/cache models and scheduling |
| `tilesight/util/` | Shared utilities |
| `examples/` | Runnable modeling examples |
| `micro_benchmark/` | CUDA/HIP probes and communication adapters |

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
```

Generated measurements and binaries remain local. [MIT license](LICENSE).
