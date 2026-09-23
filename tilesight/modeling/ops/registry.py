"""Semantic operation registry and built-in elementwise/reduce/GEMM factories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..errors import ModelingValidationError
from ..ir import validate_identifier
from .schema import (
    DTypeSpec,
    ElementwiseOpSpec,
    EngineSpec,
    GemmOpSpec,
    ReduceOpSpec,
    ReductionAxisSpec,
    ScalarWorkSpec,
    TensorOperandSpec,
    TensorResultSpec,
    as_dtype,
    concrete_elements,
    is_semantic_op,
)


Factory = Callable[..., Any]


@dataclass(frozen=True)
class OpDefinition:
    op_id: str
    family: str
    factory: Factory
    version: str = "semantic-v1"
    aliases: Tuple[str, ...] = tuple()

    def __post_init__(self) -> None:
        validate_identifier(self.op_id, "registry op_id")
        validate_identifier(self.family, "registry family")
        validate_identifier(self.version, "registry version")
        if not callable(self.factory):
            raise ModelingValidationError("registry factory must be callable")
        aliases = tuple(validate_identifier(x, "registry alias") for x in self.aliases)
        if len(aliases) != len(set(aliases)):
            raise ModelingValidationError("registry aliases must be unique")
        object.__setattr__(self, "aliases", aliases)


class SemanticOpRegistry:
    """Small explicit registry; duplicate IDs/aliases fail closed."""

    def __init__(self) -> None:
        self._definitions = {}  # type: Dict[str, OpDefinition]
        self._aliases = {}  # type: Dict[str, str]

    def register(self, definition: OpDefinition) -> None:
        if not isinstance(definition, OpDefinition):
            raise ModelingValidationError("register expects OpDefinition")
        names = (definition.op_id,) + definition.aliases
        occupied = set(self._definitions) | set(self._aliases)
        collision = sorted(set(names) & occupied)
        if collision:
            raise ModelingValidationError("duplicate registry names %s" % collision)
        self._definitions[definition.op_id] = definition
        for alias in definition.aliases:
            self._aliases[alias] = definition.op_id

    def resolve(self, op_id: str) -> OpDefinition:
        name = validate_identifier(op_id, "registry lookup")
        canonical = self._aliases.get(name, name)
        try:
            return self._definitions[canonical]
        except KeyError as error:
            raise ModelingValidationError("unknown semantic op %r" % op_id) from error

    def create(self, op_id: str, **kwargs: Any) -> Any:
        definition = self.resolve(op_id)
        value = definition.factory(op_id=definition.op_id, **kwargs)
        if not is_semantic_op(value):
            raise ModelingValidationError("semantic op factory returned an invalid object")
        if value.family != definition.family:
            raise ModelingValidationError("semantic op factory family mismatch")
        return value

    def definitions(self) -> Tuple[OpDefinition, ...]:
        return tuple(self._definitions[name] for name in sorted(self._definitions))


def _dtype_at(values: Optional[Sequence[Any]], index: int) -> DTypeSpec:
    if values is None:
        return DTypeSpec()
    if index >= len(values):
        raise ModelingValidationError("storage_dtypes needs one entry per operand")
    return as_dtype(values[index])


def make_elementwise(
    *,
    op_id: str = "elementwise.generic",
    output_shape: Sequence[Any],
    input_shapes: Sequence[Sequence[Any]],
    op_classes: Sequence[str] = ("add",),
    input_storage_dtypes: Optional[Sequence[Any]] = None,
    result_storage_dtype: Any = None,
    compute_dtype: Any = None,
    accumulation_dtype: Any = None,
    engine: Any = "cuda",
    semantic_flops_per_op: Optional[Sequence[float]] = None,
    issue_equivalent_flops_per_op: Optional[Sequence[float]] = None,
    attrs: Any = None,
) -> ElementwiseOpSpec:
    """Build arbitrary unary/binary/broadcast elementwise semantics."""

    result = TensorResultSpec("out", tuple(output_shape), as_dtype(result_storage_dtype))
    operands = []
    for index, shape in enumerate(input_shapes):
        shape = tuple(shape)
        access = "identity" if shape == result.shape else "broadcast"
        operands.append(
            TensorOperandSpec(
                "in%d" % index,
                shape,
                _dtype_at(input_storage_dtypes, index),
                access=access,
            )
        )
    if input_storage_dtypes is not None and len(input_storage_dtypes) != len(operands):
        raise ModelingValidationError("storage_dtypes needs one entry per operand")
    classes = tuple(op_classes)
    if not classes:
        raise ModelingValidationError("elementwise needs an op class")
    semantic = (
        tuple(float(x) for x in semantic_flops_per_op)
        if semantic_flops_per_op is not None
        else tuple(1.0 for _ in classes)
    )
    issue = (
        tuple(float(x) for x in issue_equivalent_flops_per_op)
        if issue_equivalent_flops_per_op is not None
        else semantic
    )
    if len(semantic) != len(classes) or len(issue) != len(classes):
        raise ModelingValidationError("elementwise work weights must match op_classes")
    dtype = as_dtype(compute_dtype)
    count = float(concrete_elements(result.shape))
    operations = tuple(
        ScalarWorkSpec(
            op_class,
            count,
            engine=engine,
            dtype=dtype,
            semantic_flops_per_op=semantic[index],
            issue_equivalent_flops_per_op=issue[index],
        )
        for index, op_class in enumerate(classes)
    )
    return ElementwiseOpSpec(
        op_id=op_id,
        operands=tuple(operands),
        result=result,
        operations=operations,
        compute_dtype=dtype,
        accumulation_dtype=as_dtype(accumulation_dtype),
        attrs=attrs,
    )


def make_reduce(
    *,
    op_id: str = "reduce.generic",
    input_shape: Sequence[Any],
    output_shape: Sequence[Any],
    reduction_axes: Sequence[Tuple[int, int, int]],
    reducer: str = "sum",
    input_storage_dtype: Any = None,
    result_storage_dtype: Any = None,
    compute_dtype: Any = None,
    accumulation_dtype: Any = None,
    map_op_classes: Sequence[str] = tuple(),
    engine: Any = "cuda",
    keepdims: bool = False,
    attrs: Any = None,
) -> ReduceOpSpec:
    """Build map-reduce work; each axis tuple is (axis, extent, tile_step)."""

    operand = TensorOperandSpec(
        "in0",
        tuple(input_shape),
        as_dtype(input_storage_dtype),
        access="reduction",
    )
    result = TensorResultSpec("out", tuple(output_shape), as_dtype(result_storage_dtype))
    axes = tuple(
        ReductionAxisSpec(axis, extent, step, (operand.name,))
        for axis, extent, step in reduction_axes
    )
    dtype = as_dtype(compute_dtype)
    output_elements = concrete_elements(result.shape)
    reduction_elements = 1
    for axis in axes:
        reduction_elements *= axis.extent
    work = []
    for op_class in map_op_classes:
        work.append(
            ScalarWorkSpec(
                op_class,
                count=float(output_elements * reduction_elements),
                engine=engine,
                dtype=dtype,
                semantic_flops_per_op=1.0,
                issue_equivalent_flops_per_op=1.0,
            )
        )
    # A reduction combines R inputs with max(R-1, 0) reducer operations.
    work.append(
        ScalarWorkSpec(
            reducer,
            count=float(output_elements * max(reduction_elements - 1, 0)),
            engine=engine,
            dtype=dtype,
            semantic_flops_per_op=1.0 if reducer in ("sum", "product") else 0.0,
            issue_equivalent_flops_per_op=1.0,
        )
    )
    return ReduceOpSpec(
        op_id=op_id,
        operands=(operand,),
        result=result,
        reduction_axes=axes,
        reducer=reducer,
        work=tuple(work),
        compute_dtype=dtype,
        accumulation_dtype=as_dtype(accumulation_dtype),
        keepdims=keepdims,
        attrs=attrs,
    )


def make_gemm(
    *,
    op_id: str = "gemm.dense",
    m: int,
    n: int,
    k: int,
    batch: int = 1,
    a_storage_dtype: Any = None,
    b_storage_dtype: Any = None,
    result_storage_dtype: Any = None,
    compute_dtype: Any = None,
    accumulation_dtype: Any = None,
    engine: Any = "tensor",
    attrs: Any = None,
) -> GemmOpSpec:
    return GemmOpSpec(
        op_id=op_id,
        m=m,
        n=n,
        k=k,
        batch=batch,
        a_storage_dtype=as_dtype(a_storage_dtype),
        b_storage_dtype=as_dtype(b_storage_dtype),
        result_storage_dtype=as_dtype(result_storage_dtype),
        compute_dtype=as_dtype(compute_dtype),
        accumulation_dtype=as_dtype(accumulation_dtype),
        engine=engine,
        attrs=attrs,
    )


def default_registry() -> SemanticOpRegistry:
    registry = SemanticOpRegistry()
    registry.register(
        OpDefinition(
            "elementwise.generic",
            "elementwise",
            make_elementwise,
            aliases=("elementwise",),
        )
    )
    registry.register(
        OpDefinition("reduce.generic", "reduce", make_reduce, aliases=("reduce",))
    )
    registry.register(
        OpDefinition("gemm.dense", "gemm", make_gemm, aliases=("gemm",))
    )
    return registry


DEFAULT_OP_REGISTRY = default_registry()


__all__ = [
    "DEFAULT_OP_REGISTRY",
    "OpDefinition",
    "SemanticOpRegistry",
    "default_registry",
    "make_elementwise",
    "make_gemm",
    "make_reduce",
]
