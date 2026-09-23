"""Tests for the independent semantic op registry and throughput binder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tilesight.modeling.errors import ModelingValidationError
from tilesight.modeling.ops import (
    DEFAULT_OP_REGISTRY,
    DTypeSpec,
    EngineSpec,
    LegacyCompatibilityPolicy,
    OpDefinition,
    SemanticOpRegistry,
    ThroughputFallbackPolicy,
    ThroughputRequest,
    UnsupportedThroughputError,
    bind_work_throughputs,
    make_elementwise,
    make_gemm,
    make_reduce,
    resolve_throughput,
)


def _arch():
    return SimpleNamespace(
        fp16_cuda_core_flops=20.0,
        fp32_cuda_core_flops=10.0,
        fp64_cuda_core_flops=5.0,
        int32_cuda_core_flops=7.0,
        fp16_tensor_flops=100.0,
        fp8_tensor_flops=200.0,
        int8_tensor_flops=180.0,
        sfu_flops=3.0,
    )


def test_unspecified_dtype_defaults_to_fp32_but_storage_compute_accum_are_separate():
    op = make_elementwise(
        output_shape=(4, 8),
        input_shapes=((4, 8), (8,)),
        input_storage_dtypes=("fp16", "bf16"),
        result_storage_dtype="bf16",
    )
    assert [item.storage_dtype.name for item in op.operands] == ["fp16", "bf16"]
    assert op.result.storage_dtype.name == "bf16"
    assert op.compute_dtype.name == "fp32"
    assert op.accumulation_dtype.name == "fp32"
    assert op.operations[0].dtype.name == "fp32"
    assert op.operands[1].access == "broadcast"
    hash(op)


def test_vector_is_only_an_alias_for_cuda_and_never_selects_tensor_throughput():
    op = make_elementwise(
        output_shape=(10,),
        input_shapes=((10,),),
        engine="vector",
    )
    engine = op.operations[0].engine
    assert engine.requested == "vector"
    assert engine.name == "cuda"
    assert engine.alias_used
    resolution = resolve_throughput(
        _arch(), ThroughputRequest(engine, DTypeSpec("fp32"), "add")
    )
    assert resolution.capacity_per_s == 10.0
    assert resolution.source == "arch.fp32_cuda_core_flops"
    assert dict(resolution.provenance)["requested_engine"] == "vector"


def test_registry_creates_typed_elementwise_reduce_and_gemm_specs():
    elementwise = DEFAULT_OP_REGISTRY.create(
        "elementwise",
        output_shape=(8,),
        input_shapes=((8,),),
        op_classes=("mul",),
    )
    reduce = DEFAULT_OP_REGISTRY.create(
        "reduce",
        input_shape=(4, 8),
        output_shape=(4,),
        reduction_axes=((1, 8, 4),),
        reducer="sum",
        map_op_classes=("mul",),
    )
    gemm = DEFAULT_OP_REGISTRY.create("gemm", m=16, n=32, k=64)
    assert elementwise.family == "elementwise"
    assert reduce.family == "reduce"
    assert reduce.reduction_iterations == 2
    assert [item.count for item in reduce.work_items] == [32.0, 28.0]
    assert gemm.family == "gemm"
    assert gemm.semantic_flops == 2 * 16 * 32 * 64
    assert gemm.engine.name == "tensor"
    assert gemm.compute_dtype.name == "fp32"
    assert gemm.accumulation_dtype.name == "fp32"
    hash(reduce)
    hash(gemm)


def test_registry_duplicate_names_fail_closed():
    registry = SemanticOpRegistry()
    definition = OpDefinition("x", "elementwise", lambda **_kwargs: None, aliases=("y",))
    registry.register(definition)
    with pytest.raises(ModelingValidationError, match="duplicate"):
        registry.register(OpDefinition("y", "elementwise", lambda **_kwargs: None))


def test_same_byte_width_does_not_create_a_silent_dtype_fallback():
    arch = _arch()
    bf16 = resolve_throughput(
        arch,
        ThroughputRequest(EngineSpec("cuda"), DTypeSpec("bf16"), "add"),
    )
    assert not bf16.supported
    assert bf16.fallback == "unsupported"
    with pytest.raises(UnsupportedThroughputError, match="bf16"):
        bf16.require()

    explicit = resolve_throughput(
        arch,
        ThroughputRequest(EngineSpec("cuda"), DTypeSpec("bf16"), "add"),
        ThroughputFallbackPolicy(dtype_aliases=(("bf16", "fp16"),)),
    )
    assert explicit.supported
    assert explicit.capacity_per_s == arch.fp16_cuda_core_flops
    assert explicit.fallback == "dtype_alias:bf16->fp16"
    assert dict(explicit.provenance)["fallback_dtype"] == "fp16"


def test_tensor_sfu_and_explicit_capacity_paths_have_provenance():
    arch = _arch()
    tensor = resolve_throughput(
        arch,
        ThroughputRequest(EngineSpec("tensor"), DTypeSpec("fp8_e4m3"), "mma"),
    )
    assert tensor.capacity_per_s == 200.0
    assert tensor.fallback == "none"

    sfu_request = ThroughputRequest(EngineSpec("sfu"), DTypeSpec("fp32"), "exp")
    assert not resolve_throughput(arch, sfu_request).supported
    sfu = resolve_throughput(
        arch,
        sfu_request,
        ThroughputFallbackPolicy(allow_dtype_agnostic_sfu=True),
    )
    assert sfu.capacity_per_s == 3.0
    assert sfu.fallback == "dtype_agnostic_sfu"
    assert dict(sfu.provenance)["confidence"] == "legacy_dtype_agnostic"

    explicit = resolve_throughput(
        SimpleNamespace(),
        ThroughputRequest(EngineSpec("cuda"), DTypeSpec("fp32"), "divide"),
        ThroughputFallbackPolicy(
            explicit_capacities=(("cuda.fp32.divide", 2.5),)
        ),
    )
    assert explicit.capacity_per_s == 2.5
    assert explicit.fallback == "explicit_capacity"


def test_work_binding_retains_unsupported_items_instead_of_guessing():
    op = make_elementwise(
        output_shape=(8,),
        input_shapes=((8,),),
        op_classes=("add",),
        compute_dtype="bf16",
        engine="vector",
    )
    bindings = bind_work_throughputs(op, _arch())
    assert len(bindings) == 1
    assert bindings[0].service_time_s is None
    assert not bindings[0].resolution.supported

    bindings = bind_work_throughputs(
        op,
        _arch(),
        ThroughputFallbackPolicy(dtype_aliases=(("bf16", "fp16"),)),
    )
    assert bindings[0].service_time_s == pytest.approx(8.0 / 20.0)
    assert bindings[0].resolution.fallback == "dtype_alias:bf16->fp16"


def test_legacy_policy_reproduces_thread_bucket_and_fma_equivalent_formulas():
    policy = LegacyCompatibilityPolicy()
    assert policy.thread_overhead(32) == 1.0
    assert policy.thread_overhead(33) == pytest.approx(128.0 / 33.0)
    assert policy.thread_overhead(129) == pytest.approx(256.0 / 129.0)
    assert policy.elementwise_issue_equivalent_flops(100, 3, 32) == 600
    assert policy.reduce_issue_equivalent_flops(16, 8, 32) == 256
    assert policy.collective_issue_equivalent_flops(16, 128, 128) == 224
    provenance = dict(policy.provenance())
    assert provenance["profile"] == "legacy_tilesight_v0"
    assert provenance["semantic_default"] is False


def test_invalid_broadcast_and_non_tensor_gemm_are_rejected():
    with pytest.raises(ModelingValidationError, match="broadcast"):
        make_elementwise(output_shape=(4, 8), input_shapes=((3, 8),))
    with pytest.raises(ModelingValidationError, match="tensor engine"):
        make_gemm(m=8, n=8, k=8, engine="vector")
