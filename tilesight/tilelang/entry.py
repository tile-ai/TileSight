"""Resolve a kernel request (file, factory, specialization) into one TileLang PrimFunc.

Rules: the named symbol is either a kernel object or a factory called with
``factory_args``; the result is a PrimFunc, an IRModule with one PrimFunc (or
the named one), an object with ``get_tir(**specialization)``, or an object
exposing ``prim_func``. Anything else is refused.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Tuple

from .errors import Refusal


@dataclass(frozen=True)
class KernelRequest:
    file: str
    function: str
    factory_args: Mapping[str, Any] = field(default_factory=dict)
    specialization: Mapping[str, Any] = field(default_factory=dict)
    target_arch: str = "sm_90a"
    pass_configs: Optional[Mapping[str, Any]] = None
    bindings: Mapping[str, int] = field(default_factory=dict)
    prim_func_name: Optional[str] = None


@dataclass(frozen=True)
class ResolvedKernel:
    prim_func: Any
    pass_configs: Tuple[Tuple[str, Any], ...]
    pass_configs_source: str
    source_sha256: str
    name: str


@contextlib.contextmanager
def _sys_path(path: Path) -> Iterator[None]:
    text = str(path)
    sys.path.insert(0, text)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(text)


def _load_module(path: Path) -> Tuple[Any, str]:
    if not path.is_file():
        raise Refusal("ENTRY_FILE_NOT_FOUND", "kernel file does not exist: %s" % path)
    source = path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    name = "_tilesight_kernel_%s" % digest[:20]
    if name in sys.modules:
        return sys.modules[name], digest
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Refusal("ENTRY_MODULE_LOAD_UNSUPPORTED", "cannot import %s" % path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # TileLang decorators keep FFI objects owned by the module
    try:
        with _sys_path(path.parent):
            spec.loader.exec_module(module)
    except Exception as error:
        sys.modules.pop(name, None)
        raise Refusal("ENTRY_IMPORT_FAILED", "importing %s failed: %s: %s" % (path, type(error).__name__, error)) from error
    return module, digest


def _symbol(module: Any, dotted: str, path: Path) -> Any:
    value = module
    for part in dotted.split("."):
        if not part or not hasattr(value, part):
            raise Refusal("ENTRY_SYMBOL_NOT_FOUND", "symbol %r was not found in %s" % (dotted, path))
        value = getattr(value, part)
    return value


def _is_primfunc(value: Any) -> bool:
    from tilelang import tvm
    return isinstance(value, tvm.tirx.PrimFunc) if hasattr(tvm, "tirx") else type(value).__name__ == "PrimFunc"


def _is_module(value: Any) -> bool:
    from tilelang import tvm
    return isinstance(value, tvm.IRModule)


def _kernel_like(value: Any) -> bool:
    return _is_primfunc(value) or _is_module(value) or hasattr(value, "get_tir") or hasattr(value, "prim_func")


def _call_factory(value: Any, args: Mapping[str, Any]) -> Any:
    if _kernel_like(value):
        if args:
            raise Refusal("ENTRY_FACTORY_ARGS_UNUSED", "factory_args were given but the symbol is already a kernel")
        return value
    if not callable(value):
        raise Refusal("ENTRY_SYMBOL_NOT_CALLABLE", "symbol is neither a kernel nor a factory")
    try:
        inspect.signature(value).bind(**args)
    except (TypeError, ValueError) as error:
        raise Refusal("ENTRY_FACTORY_ARGUMENT_MISMATCH", "factory arguments do not match: %s" % error) from error
    try:
        return value(**args)
    except Exception as error:
        raise Refusal("ENTRY_FACTORY_FAILED", "kernel factory failed: %s: %s" % (type(error).__name__, error)) from error


def _select(value: Any, name: Optional[str]) -> Any:
    if _is_primfunc(value):
        return value
    if _is_module(value):
        functions = {gv.name_hint: f for gv, f in value.functions.items() if _is_primfunc(f)}
        if name is not None:
            if name not in functions:
                raise Refusal("ENTRY_PRIMFUNC_NOT_FOUND", "IRModule has no PrimFunc %r" % name)
            return functions[name]
        if len(functions) != 1:
            raise Refusal("ENTRY_PRIMFUNC_AMBIGUOUS", "IRModule has %d PrimFuncs; name one" % len(functions))
        return next(iter(functions.values()))
    raise Refusal("ENTRY_TIR_UNAVAILABLE", "object is neither a PrimFunc nor an IRModule")


def resolve(request: KernelRequest) -> ResolvedKernel:
    path = Path(request.file).expanduser().resolve()
    module, digest = _load_module(path)
    kernel = _call_factory(_symbol(module, request.function, path), dict(request.factory_args))
    specialization = dict(request.specialization)
    if hasattr(kernel, "get_tir"):
        try:
            tir = kernel.get_tir(**specialization)
        except Exception as error:
            raise Refusal("ENTRY_GET_TIR_FAILED", "get_tir failed: %s: %s" % (type(error).__name__, error)) from error
    elif _is_primfunc(kernel) or _is_module(kernel):
        if specialization:
            raise Refusal("ENTRY_SPECIALIZATION_UNUSED", "specialization given for an already materialized PrimFunc")
        tir = kernel
    else:
        if specialization:
            raise Refusal("ENTRY_SPECIALIZATION_UNUSED", "specialization given but the kernel only exposes prim_func")
        tir = kernel.prim_func
    prim_func = _select(tir, request.prim_func_name)
    if request.pass_configs is not None:
        configs, source = dict(request.pass_configs), "request"
    elif isinstance(getattr(kernel, "pass_configs", None), Mapping):
        configs, source = dict(kernel.pass_configs), "kernel object"
    else:
        configs, source = {}, "default"
    return ResolvedKernel(prim_func, tuple(sorted((str(k), v) for k, v in configs.items())), source, digest,
                          "%s:%s" % (path.name, request.function))
