"""Frozen architecture-neutral semantic operation schemas.

The schemas deliberately separate tensor storage dtype, arithmetic dtype, and
accumulator dtype.  They describe work, not cache traffic, occupancy, waves,
or launch topology.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import reduce
from operator import mul
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from ..errors import ModelingValidationError
from ..ir import freeze_attrs, freeze_value, validate_identifier


_DTYPE_ALIASES = {
    "float": "fp32",
    "float32": "fp32",
    "f32": "fp32",
    "float16": "fp16",
    "half": "fp16",
    "f16": "fp16",
    "bfloat16": "bf16",
    "float64": "fp64",
    "double": "fp64",
    "f64": "fp64",
    "fp8": "fp8_e4m3",
    "e4m3": "fp8_e4m3",
    "e5m2": "fp8_e5m2",
    "i8": "int8",
    "i16": "int16",
    "i32": "int32",
    "i64": "int64",
    "u8": "uint8",
    "boolean": "bool",
}

_DTYPE_BYTES = {
    "bool": 1,
    "fp8_e4m3": 1,
    "fp8_e5m2": 1,
    "int8": 1,
    "uint8": 1,
    "bf16": 2,
    "fp16": 2,
    "int16": 2,
    "fp32": 4,
    "tf32": 4,
    "int32": 4,
    "fp64": 8,
    "int64": 8,
}

_ENGINE_ALIASES = {
    "vector": "cuda",
    "cuda_core": "cuda",
    "cuda-core": "cuda",
    "tensor_core": "tensor",
    "tensor-core": "tensor",
}

_ENGINES = frozenset(("cuda", "tensor", "sfu", "copy"))


def _positive_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result <= 0.0:
        raise ModelingValidationError("%s must be finite and positive" % label)
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result < 0.0:
        raise ModelingValidationError("%s must be finite and non-negative" % label)
    return result


def _shape(value: Sequence[Any], label: str) -> Tuple[Any, ...]:
    try:
        result = tuple(freeze_value(item) for item in value)
    except TypeError as error:
        raise ModelingValidationError("%s must be a shape sequence" % label) from error
    if not result:
        raise ModelingValidationError("%s must not be empty" % label)
    for extent in result:
        if isinstance(extent, int) and not isinstance(extent, bool) and extent <= 0:
            raise ModelingValidationError("%s concrete extents must be positive" % label)
    return result


def concrete_elements(shape: Sequence[Any]) -> int:
    """Return concrete element count or fail instead of guessing symbols."""

    result = 1
    for extent in shape:
        if not isinstance(extent, int) or isinstance(extent, bool) or extent <= 0:
            raise ModelingValidationError(
                "element count requires concrete positive integer extents"
            )
        result *= extent
    return result


@dataclass(frozen=True)
class DTypeSpec:
    """Canonical dtype identity; unspecified values normalize to FP32."""

    name: Optional[str] = "fp32"

    def __post_init__(self) -> None:
        value = "fp32" if self.name in (None, "", "unspecified") else str(self.name).lower()
        value = _DTYPE_ALIASES.get(value, value)
        if value not in _DTYPE_BYTES:
            raise ModelingValidationError("unsupported dtype %r" % self.name)
        object.__setattr__(self, "name", value)

    @property
    def storage_bytes(self) -> int:
        return _DTYPE_BYTES[self.name]


def as_dtype(value: Any) -> DTypeSpec:
    return value if isinstance(value, DTypeSpec) else DTypeSpec(value)


@dataclass(frozen=True)
class EngineSpec:
    """Canonical engine while retaining the user-facing spelling."""

    requested: str = "cuda"
    name: str = field(init=False)

    def __post_init__(self) -> None:
        requested = validate_identifier(self.requested, "engine").lower()
        canonical = _ENGINE_ALIASES.get(requested, requested)
        if canonical not in _ENGINES:
            raise ModelingValidationError("unsupported engine %r" % self.requested)
        object.__setattr__(self, "requested", requested)
        object.__setattr__(self, "name", canonical)

    @property
    def alias_used(self) -> bool:
        return self.requested != self.name


def as_engine(value: Any) -> EngineSpec:
    return value if isinstance(value, EngineSpec) else EngineSpec(str(value))


@dataclass(frozen=True)
class TensorOperandSpec:
    name: str
    shape: Tuple[Any, ...]
    storage_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    access: str = "identity"
    index_map: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.name, "operand name")
        object.__setattr__(self, "shape", _shape(self.shape, "operand shape"))
        object.__setattr__(self, "storage_dtype", as_dtype(self.storage_dtype))
        if self.access not in ("identity", "broadcast", "affine", "reduction"):
            raise ModelingValidationError("unsupported operand access %r" % self.access)
        object.__setattr__(self, "index_map", freeze_attrs(self.index_map))


@dataclass(frozen=True)
class TensorResultSpec:
    name: str
    shape: Tuple[Any, ...]
    storage_dtype: DTypeSpec = field(default_factory=DTypeSpec)

    def __post_init__(self) -> None:
        validate_identifier(self.name, "result name")
        object.__setattr__(self, "shape", _shape(self.shape, "result shape"))
        object.__setattr__(self, "storage_dtype", as_dtype(self.storage_dtype))


@dataclass(frozen=True)
class ScalarWorkSpec:
    """One arithmetic/SFU work class with honest semantic and issue units."""

    op_class: str
    count: float
    engine: EngineSpec = field(default_factory=EngineSpec)
    dtype: DTypeSpec = field(default_factory=DTypeSpec)
    semantic_flops_per_op: float = 0.0
    issue_equivalent_flops_per_op: float = 0.0
    attrs: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.op_class, "scalar op class")
        object.__setattr__(self, "count", _nonnegative_float(self.count, "work count"))
        object.__setattr__(self, "engine", as_engine(self.engine))
        object.__setattr__(self, "dtype", as_dtype(self.dtype))
        object.__setattr__(
            self,
            "semantic_flops_per_op",
            _nonnegative_float(self.semantic_flops_per_op, "semantic_flops_per_op"),
        )
        object.__setattr__(
            self,
            "issue_equivalent_flops_per_op",
            _nonnegative_float(
                self.issue_equivalent_flops_per_op,
                "issue_equivalent_flops_per_op",
            ),
        )
        object.__setattr__(self, "attrs", freeze_attrs(self.attrs))

    @property
    def semantic_flops(self) -> float:
        return self.count * self.semantic_flops_per_op

    @property
    def issue_equivalent_flops(self) -> float:
        return self.count * self.issue_equivalent_flops_per_op


def _broadcastable(input_shape: Tuple[Any, ...], output_shape: Tuple[Any, ...]) -> bool:
    if len(input_shape) > len(output_shape):
        return False
    padded = (1,) * (len(output_shape) - len(input_shape)) + input_shape
    for source, target in zip(padded, output_shape):
        if source == 1 or source == target:
            continue
        if not isinstance(source, int) or not isinstance(target, int):
            # Symbolic equality beyond object equality needs an external solver.
            return False
        return False
    return True


class SemanticOpSpec:
    """Marker base for frozen semantic operation specifications."""


@dataclass(frozen=True)
class ElementwiseOpSpec(SemanticOpSpec):
    op_id: str
    operands: Tuple[TensorOperandSpec, ...]
    result: TensorResultSpec
    operations: Tuple[ScalarWorkSpec, ...]
    compute_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    accumulation_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    attrs: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.op_id, "elementwise op_id")
        operands = tuple(self.operands)
        operations = tuple(self.operations)
        if not operands or not all(isinstance(item, TensorOperandSpec) for item in operands):
            raise ModelingValidationError("elementwise operands must be nonempty")
        if not isinstance(self.result, TensorResultSpec):
            raise ModelingValidationError("elementwise result must be TensorResultSpec")
        if not operations or not all(isinstance(item, ScalarWorkSpec) for item in operations):
            raise ModelingValidationError("elementwise operations must be nonempty")
        for operand in operands:
            if not _broadcastable(operand.shape, self.result.shape):
                raise ModelingValidationError(
                    "operand %s does not broadcast to the result" % operand.name
                )
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "operations", operations)
        object.__setattr__(self, "compute_dtype", as_dtype(self.compute_dtype))
        object.__setattr__(self, "accumulation_dtype", as_dtype(self.accumulation_dtype))
        object.__setattr__(self, "attrs", freeze_attrs(self.attrs))

    @property
    def family(self) -> str:
        return "elementwise"

    @property
    def work_items(self) -> Tuple[ScalarWorkSpec, ...]:
        return self.operations


@dataclass(frozen=True)
class ReductionAxisSpec:
    axis: int
    extent: int
    tile_step: int
    operand_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.axis, int) or isinstance(self.axis, bool) or self.axis < 0:
            raise ModelingValidationError("reduction axis must be non-negative")
        for value, label in ((self.extent, "extent"), (self.tile_step, "tile_step")):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ModelingValidationError("reduction %s must be positive" % label)
        names = tuple(validate_identifier(x, "reduction operand") for x in self.operand_names)
        if not names:
            raise ModelingValidationError("reduction axis needs an operand")
        object.__setattr__(self, "operand_names", names)

    @property
    def iterations(self) -> int:
        return int(math.ceil(self.extent / self.tile_step))


@dataclass(frozen=True)
class ReduceOpSpec(SemanticOpSpec):
    op_id: str
    operands: Tuple[TensorOperandSpec, ...]
    result: TensorResultSpec
    reduction_axes: Tuple[ReductionAxisSpec, ...]
    reducer: str = "sum"
    work: Tuple[ScalarWorkSpec, ...] = field(default_factory=tuple)
    compute_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    accumulation_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    keepdims: bool = False
    attrs: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.op_id, "reduce op_id")
        validate_identifier(self.reducer, "reducer")
        operands = tuple(self.operands)
        axes = tuple(self.reduction_axes)
        work = tuple(self.work)
        if not operands or not all(isinstance(x, TensorOperandSpec) for x in operands):
            raise ModelingValidationError("reduce operands must be nonempty")
        if not isinstance(self.result, TensorResultSpec):
            raise ModelingValidationError("reduce result must be TensorResultSpec")
        if not axes or not all(isinstance(x, ReductionAxisSpec) for x in axes):
            raise ModelingValidationError("reduce needs reduction axes")
        known = {operand.name for operand in operands}
        if any(name not in known for axis in axes for name in axis.operand_names):
            raise ModelingValidationError("reduction axis references unknown operand")
        if not work or not all(isinstance(x, ScalarWorkSpec) for x in work):
            raise ModelingValidationError("reduce work must be nonempty")
        if not isinstance(self.keepdims, bool):
            raise ModelingValidationError("keepdims must be bool")
        object.__setattr__(self, "operands", operands)
        object.__setattr__(self, "reduction_axes", axes)
        object.__setattr__(self, "work", work)
        object.__setattr__(self, "compute_dtype", as_dtype(self.compute_dtype))
        object.__setattr__(self, "accumulation_dtype", as_dtype(self.accumulation_dtype))
        object.__setattr__(self, "attrs", freeze_attrs(self.attrs))

    @property
    def family(self) -> str:
        return "reduce"

    @property
    def reduction_iterations(self) -> int:
        return reduce(mul, (axis.iterations for axis in self.reduction_axes), 1)

    @property
    def work_items(self) -> Tuple[ScalarWorkSpec, ...]:
        return self.work


@dataclass(frozen=True)
class GemmOpSpec(SemanticOpSpec):
    op_id: str = "gemm.dense"
    m: int = 1
    n: int = 1
    k: int = 1
    batch: int = 1
    a_storage_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    b_storage_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    result_storage_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    compute_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    accumulation_dtype: DTypeSpec = field(default_factory=DTypeSpec)
    engine: EngineSpec = field(default_factory=lambda: EngineSpec("tensor"))
    attrs: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        validate_identifier(self.op_id, "GEMM op_id")
        for name in ("m", "n", "k", "batch"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ModelingValidationError("GEMM %s must be positive" % name)
        for name in (
            "a_storage_dtype",
            "b_storage_dtype",
            "result_storage_dtype",
            "compute_dtype",
            "accumulation_dtype",
        ):
            object.__setattr__(self, name, as_dtype(getattr(self, name)))
        engine = as_engine(self.engine)
        if engine.name != "tensor":
            raise ModelingValidationError("GEMM v1 requires explicit tensor engine")
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "attrs", freeze_attrs(self.attrs))

    @property
    def family(self) -> str:
        return "gemm"

    @property
    def semantic_flops(self) -> float:
        return float(2 * self.batch * self.m * self.n * self.k)

    @property
    def work_items(self) -> Tuple[ScalarWorkSpec, ...]:
        return (
            ScalarWorkSpec(
                "mma",
                count=float(self.batch * self.m * self.n * self.k),
                engine=self.engine,
                dtype=self.compute_dtype,
                semantic_flops_per_op=2.0,
                issue_equivalent_flops_per_op=2.0,
                attrs={"accumulation_dtype": self.accumulation_dtype.name},
            ),
        )


def is_semantic_op(value: Any) -> bool:
    return isinstance(value, SemanticOpSpec)


__all__ = [
    "DTypeSpec",
    "ElementwiseOpSpec",
    "EngineSpec",
    "GemmOpSpec",
    "ReduceOpSpec",
    "ReductionAxisSpec",
    "ScalarWorkSpec",
    "SemanticOpSpec",
    "TensorOperandSpec",
    "TensorResultSpec",
    "as_dtype",
    "as_engine",
    "concrete_elements",
    "is_semantic_op",
]
