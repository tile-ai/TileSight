"""One official TileLang lowering, observed with public APIs only.

The kernel is lowered once with ``tilelang.lower(..., enable_device_compile=False)``
(no NVCC, no GPU). A TVM pass instrument keeps the module right after
``PipelinePlanning``: tile operations are still tile operations, while the
compiler has already decided warp-specialized roles, ring buffers and pipeline
stages. The final device module supplies launch facts (threads, dynamic shared
memory) that only the complete lowering knows.

Every failure is a refusal with a code. In particular a lowering that raises
anywhere is refused: a kernel that does not compile does not run, so it has no
time to estimate.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from .errors import Refusal

PLANNED_PASS = "PipelinePlanning"
SUPPORTED_TILELANG = ("0.1.12", "0.1.14")


@dataclass(frozen=True)
class DeviceFacts:
    """Launch facts of the single device kernel produced by the lowering."""

    symbol: str
    grid: Tuple[Any, ...]          # blockIdx.x/y/z extents; may still contain shape variables
    threads: Tuple[int, int, int]  # threadIdx.x/y/z
    dynamic_shared_bytes: Optional[int]
    mechanisms: Tuple[str, ...]    # instruction families seen in the generated source

    @property
    def thread_count(self) -> int:
        x, y, z = self.threads
        return x * y * z


@dataclass(frozen=True)
class Capture:
    planned: Any                   # tirx.PrimFunc after PipelinePlanning (translation input)
    device: DeviceFacts
    target_arch: str
    pass_configs: Tuple[Tuple[str, Any], ...]
    compiler: Tuple[Tuple[str, str], ...]
    passes: Tuple[str, ...] = field(repr=False, default=())
    planned_sha256: str = ""
    kernel_source: str = field(repr=False, default="")


@lru_cache(maxsize=1)
def compiler_identity() -> Tuple[Tuple[str, str], ...]:
    """TileLang version plus a fingerprint of the installed pipeline sources and libraries."""

    import tilelang
    root = Path(tilelang.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(list(root.rglob("*.py")) + list(root.rglob("*.so"))):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return (("tilelang_version", str(tilelang.__version__)), ("tilelang_fingerprint", digest.hexdigest()))


def _check_version() -> None:
    import tilelang
    version = str(getattr(tilelang, "__version__", ""))
    if not any(version.startswith(v) for v in SUPPORTED_TILELANG):
        raise Refusal("TILELANG_VERSION_UNSUPPORTED",
                      "TileLang %s is not supported; supported versions: %s" % (version, ", ".join(SUPPORTED_TILELANG)))


def _target(arch: str) -> Any:
    from tilelang import tvm
    return tvm.target.Target({"kind": "cuda", "arch": arch})


_MECHANISMS = (
    ("tma", re.compile(r"tma_load|tma_store|cp\.async\.bulk")),
    ("cp_async", re.compile(r"cp_async|cp\.async\.(ca|cg)")),
    ("wgmma", re.compile(r"wgmma")),
    ("mma", re.compile(r"\bmma\.sync|tl::mma_sync|ptx_mma|gemm_ss|gemm_rs")),
    ("tcgen05", re.compile(r"tcgen05")),
)


def _device_facts(artifact: Any) -> DeviceFacts:
    device_mod = getattr(artifact, "device_mod", None)
    if device_mod is None:
        raise Refusal("DEVICE_MODULE_MISSING", "the lowering produced no device module")
    kernels = [(gv, f) for gv, f in device_mod.functions.items()
               if f.attrs is not None and "thread_extent" in f.attrs]
    if len(kernels) != 1:
        raise Refusal("KERNEL_COUNT_UNSUPPORTED",
                      "expected exactly one device kernel, found %d" % len(kernels))
    gv, func = kernels[0]
    extent = dict(func.attrs["thread_extent"])

    def get(name: str, default: int = 1) -> Any:
        value = extent.get(name)
        if value is None:
            return default
        return int(value) if hasattr(value, "value") or isinstance(value, int) else value

    threads = tuple(get("threadIdx." + a) for a in "xyz")
    if not all(isinstance(t, int) for t in threads):
        raise Refusal("THREADS_SYMBOLIC", "thread extents are not constant: %r" % (threads,))
    dyn = func.attrs.get("dyn_shared_memory_buf")
    source = str(getattr(artifact, "kernel_source", "") or "")
    mechanisms = tuple(name for name, pattern in _MECHANISMS if pattern.search(source))
    return DeviceFacts(
        symbol=str(func.attrs.get("global_symbol", gv.name_hint)),
        grid=tuple(get("blockIdx." + a) for a in "xyz"),
        threads=threads,  # type: ignore[arg-type]
        dynamic_shared_bytes=None if dyn is None else int(dyn),
        mechanisms=mechanisms,
    )


def capture(prim_func: Any, *, target_arch: str, pass_configs: Optional[Mapping[str, Any]] = None) -> Capture:
    """Lower ``prim_func`` once and keep the planned module and the device launch facts."""

    _check_version()
    import tilelang
    from tilelang import tvm

    configs = dict(pass_configs or {})
    planned: list = []
    passes: list = []

    @tvm.ir.instrument.pass_instrument
    class _Tap:
        def run_after_pass(self, mod, info):  # noqa: N805 - TVM instrument protocol
            name = str(info.name)
            passes.append(name)
            if name.split(".")[-1] == PLANNED_PASS:
                planned.append(mod)

    target = _target(target_arch)
    try:
        with tvm.transform.PassContext(opt_level=3, config=configs, instruments=[_Tap()]), target:
            artifact = tilelang.lower(prim_func, target=target, enable_device_compile=False)
    except Refusal:
        raise
    except Exception as error:  # the kernel does not compile: refuse, never continue on a partial lowering
        last = passes[-1] if passes else "before the first pass"
        message = (str(error).strip().splitlines() or [type(error).__name__])[-1][:300]
        raise Refusal("LOWERING_FAILED", "TileLang lowering failed after %s: %s" % (last, message)) from error
    if len(planned) != 1:
        raise Refusal("CHECKPOINT_AMBIGUOUS" if planned else "CHECKPOINT_MISSING",
                      "expected one %s snapshot, saw %d" % (PLANNED_PASS, len(planned)))
    module = planned[0]
    functions = list(module.functions.values())
    if len(functions) != 1:
        raise Refusal("MODULE_UNSUPPORTED", "expected one PrimFunc in the planned module, saw %d" % len(functions))
    text = module.script()
    return Capture(
        planned=functions[0],
        device=_device_facts(artifact),
        target_arch=target_arch,
        pass_configs=tuple(sorted((str(k), v) for k, v in configs.items())),
        compiler=compiler_identity(),
        passes=tuple(passes),
        planned_sha256=hashlib.sha256(text.encode()).hexdigest(),
        kernel_source=str(getattr(artifact, "kernel_source", "") or ""),
    )
