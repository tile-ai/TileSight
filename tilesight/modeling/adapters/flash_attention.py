"""Source-tree adapters for the audited FA3 and FA4 whole-kernel models.

The schedule frontend intentionally does not copy the calibrated whole-kernel
equations from ``tilesight/modeling/_attention``.  This adapter calls those audited
models verbatim and converts their dictionaries into the common modeling
result protocol.  Keeping the import lazy has two useful properties:

* importing :mod:`tilesight.modeling` never imports the source-tree
  FA3/FA4 model modules;
* an installed TileSight wheel fails with an actionable message when the
  repository-only case-study models are not present.

Only this adapter knows the legacy dictionary spellings.  The public executor
can therefore keep one typed result surface for GEMM, FA3, FA4, and future
frontends.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from types import ModuleType
from typing import Any, Dict, Mapping, Optional, Tuple

from ..errors import ModelingError, ModelingValidationError
from ..full_model import (
    FullModelError,
    FullModelOptions,
    FullModelResult,
    GridResult,
    IISummary,
    OccupancyResult,
    TimingBreakdown,
)
from ..ir import KernelIR, LaunchIR, PeriodicLoopIR
from ..liveness import analyze_fusion, analyze_liveness


_FAMILY_BY_KERNEL_NAME = {
    "fa3_forward": "fa3",
    "fa4_forward": "fa4",
}

_LEGACY_MODULE = {
    "fa3": "tilesight.modeling._attention.fa3_model",
    "fa4": "tilesight.modeling._attention.fa4_model",
}


class LegacyFlashAttentionModelUnavailable(ModelingError):
    """Raised when a source-tree-only audited FA model cannot be imported."""


@dataclass(frozen=True)
class FlashAttentionSpec:
    """Normalized whole-kernel inputs recovered from ``KernelIR.params``."""

    family: str
    batch: int
    query_heads: int
    sequence: int
    head_dim: int
    kv_heads: int
    causal: bool
    dtype_bytes: int
    accumulator_bytes: int
    max_util: float
    correction_frequency: float = 1.0


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ModelingValidationError("%s must be a positive integer" % label)
    return value


def _finite_fraction(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result <= 0.0 or result > 1.0:
        raise ModelingValidationError("%s must be in (0, 1]" % label)
    return result


def _param_alias(
    params: Mapping[str, Any], names: Tuple[str, ...], default: Any
) -> Any:
    present = [(name, params[name]) for name in names if name in params]
    if not present:
        return default
    first_name, first_value = present[0]
    for name, value in present[1:]:
        if value != first_value:
            raise ModelingValidationError(
                "conflicting kernel params %s=%r and %s=%r"
                % (first_name, first_value, name, value)
            )
    return first_value


def _shape_from_params(params: Mapping[str, Any]) -> Tuple[int, int, int, int]:
    shape = params.get("shape")
    scalar_names = ("batch", "query_heads", "sequence", "head_dim")
    scalar_values = tuple(params.get(name) for name in scalar_names)
    has_scalars = any(value is not None for value in scalar_values)

    if shape is None and not has_scalars:
        raise ModelingValidationError(
            "FlashAttention KernelIR.params needs shape=(batch, heads, sequence, head_dim)"
        )
    if shape is not None:
        if not isinstance(shape, (tuple, list)) or len(shape) != 4:
            raise ModelingValidationError(
                "FlashAttention shape must be (batch, heads, sequence, head_dim)"
            )
        normalized = tuple(
            _positive_int(value, "shape[%d]" % index)
            for index, value in enumerate(shape)
        )
        if has_scalars:
            for index, (name, value) in enumerate(zip(scalar_names, scalar_values)):
                if value is not None and value != normalized[index]:
                    raise ModelingValidationError(
                        "kernel param %s=%r conflicts with shape[%d]=%r"
                        % (name, value, index, normalized[index])
                    )
        return normalized  # type: ignore[return-value]

    if any(value is None for value in scalar_values):
        missing = [
            name for name, value in zip(scalar_names, scalar_values) if value is None
        ]
        raise ModelingValidationError(
            "incomplete FlashAttention scalar shape params: missing %s"
            % ", ".join(missing)
        )
    return tuple(
        _positive_int(value, name)
        for name, value in zip(scalar_names, scalar_values)
    )  # type: ignore[return-value]


def _dtype_bytes(params: Mapping[str, Any]) -> int:
    dtype = params.get("dtype", "bf16")
    inferred = {"bf16": 2, "bfloat16": 2, "fp16": 2, "float16": 2}.get(dtype)
    explicit = params.get("dtype_bytes")
    if explicit is None:
        if inferred is None:
            raise ModelingValidationError(
                "dtype %r needs an explicit positive dtype_bytes value" % (dtype,)
            )
        return inferred
    explicit = _positive_int(explicit, "dtype_bytes")
    if inferred is not None and explicit != inferred:
        raise ModelingValidationError(
            "dtype=%r conflicts with dtype_bytes=%d" % (dtype, explicit)
        )
    return explicit


def resolve_flash_attention_spec(kernel: KernelIR) -> FlashAttentionSpec:
    """Normalize an FA3/FA4 frontend program without importing case studies."""

    if not isinstance(kernel, KernelIR):
        raise ModelingValidationError("flash-attention adapter expects KernelIR")
    params = dict(kernel.params)
    inferred_family = _FAMILY_BY_KERNEL_NAME.get(kernel.name)
    family = _param_alias(params, ("model_family", "family"), inferred_family)
    if family not in ("fa3", "fa4"):
        raise ModelingValidationError(
            "flash-attention adapter requires model_family='fa3' or 'fa4'; got %r"
            % (family,)
        )
    if inferred_family is not None and inferred_family != family:
        raise ModelingValidationError(
            "kernel name %r implies %s but model_family=%r"
            % (kernel.name, inferred_family, family)
        )

    batch, query_heads, sequence, head_dim = _shape_from_params(params)
    kv_heads = _positive_int(
        _param_alias(params, ("kv_heads", "kv_h"), query_heads), "kv_heads"
    )
    if query_heads % kv_heads:
        raise ModelingValidationError(
            "query_heads=%d must be divisible by kv_heads=%d"
            % (query_heads, kv_heads)
        )

    causal_default = False if family == "fa3" else True
    causal = params.get("causal", causal_default)
    if not isinstance(causal, bool):
        raise ModelingValidationError("causal must be bool")
    accumulator_bytes = _positive_int(
        _param_alias(params, ("accumulator_bytes", "accum_bytes"), 4),
        "accumulator_bytes",
    )
    max_util = _finite_fraction(params.get("max_util", 0.85), "max_util")
    correction_frequency = _finite_fraction(
        params.get("correction_frequency", params.get("correction_freq", 1.0)),
        "correction_frequency",
    )

    return FlashAttentionSpec(
        family=family,
        batch=batch,
        query_heads=query_heads,
        sequence=sequence,
        head_dim=head_dim,
        kv_heads=kv_heads,
        causal=causal,
        dtype_bytes=_dtype_bytes(params),
        accumulator_bytes=accumulator_bytes,
        max_util=max_util,
        correction_frequency=correction_frequency,
    )


@dataclass(frozen=True)
class _ValidatedFATemplate:
    launch: LaunchIR
    loop: PeriodicLoopIR
    tile_m: int
    tile_n: int
    q_stages: int
    kv_stages: int


def _template_error(family: str, message: str) -> ModelingValidationError:
    return ModelingValidationError("%s frontend template: %s" % (family.upper(), message))


def _require_equal(family: str, label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise _template_error(
            family,
            "%s must be %r for the audited model, got %r"
            % (label, expected, actual),
        )


def _positive_tuple(value: Any, label: str, length: int) -> Tuple[int, ...]:
    try:
        result = tuple(value)
    except TypeError as error:
        raise ModelingValidationError("%s must be a sequence" % label) from error
    if len(result) != length:
        raise ModelingValidationError(
            "%s must contain exactly %d values" % (label, length)
        )
    return tuple(_positive_int(item, label) for item in result)


def _fa4_kv_stages(
    tile_m: int,
    tile_n: int,
    head_dim: int,
    q_stages: int,
    dtype_bytes: int,
) -> int:
    """Source-equivalent SM100 K/V slot budget without importing periodic code."""

    padded = int(math.ceil(head_dim / 16.0) * 16)
    smem_q = q_stages * tile_m * padded * dtype_bytes
    smem_o = q_stages * tile_m * padded * dtype_bytes
    overlap_q_o = padded == 192
    smem_q_o = max(smem_q, smem_o) if overlap_q_o else smem_q + smem_o
    slot_bytes = tile_n * padded * dtype_bytes
    stages = min((224 * 1024 - smem_q_o) // slot_bytes, 32)
    # The source exception is stated for D=192, Dv=128.  The current full FA4
    # adapter has Dv=D, so it does not trigger, but retain the general spelling
    # to make the equivalence obvious if Dv becomes a frontend parameter.
    if padded == 192 and head_dim == 128 and stages == 2:
        stages = 3
    if stages <= 0:
        raise _template_error("fa4", "SMEM budget leaves no K/V pipeline slot")
    return int(stages)


def _phase_map(family: str, loop: PeriodicLoopIR) -> Dict[str, Any]:
    phases = [phase for actor in loop.actors for phase in actor.phases]
    names = [phase.name for phase in phases]
    if len(names) != len(set(names)):
        raise _template_error(family, "phase names must be unique")
    return {phase.name: phase for phase in phases}


def _occurrences(sequence: Any) -> Tuple[Tuple[str, int], ...]:
    return tuple((item.phase.name, item.window_offset) for item in sequence)


def _validate_actor_and_resource_sequences(
    family: str,
    loop: PeriodicLoopIR,
    expected_actor_phases: Mapping[str, Tuple[str, ...]],
    expected_actor_sequences: Mapping[str, Tuple[Tuple[str, int], ...]],
    expected_resource_sequences: Mapping[str, Tuple[Tuple[str, int], ...]],
) -> Dict[str, Any]:
    actors = {actor.name: actor for actor in loop.actors}
    if len(actors) != len(loop.actors):
        raise _template_error(family, "actor names must be unique")
    _require_equal(family, "actor set", set(actors), set(expected_actor_phases))
    for name, expected_phases in expected_actor_phases.items():
        actor = actors[name]
        _require_equal(
            family,
            "actor %s phase set" % name,
            set(phase.name for phase in actor.phases),
            set(expected_phases),
        )
        _require_equal(family, "actor %s order" % name, actor.order, "issue")
        _require_equal(
            family,
            "actor %s execution_scope" % name,
            actor.execution_scope,
            "unspecified",
        )
        _require_equal(
            family,
            "actor %s execution_domain" % name,
            actor.execution_domain,
            None,
        )
        _require_equal(
            family, "actor %s serial_resource" % name, actor.serial_resource, None
        )
        _require_equal(
            family,
            "actor %s sequence" % name,
            _occurrences(actor.sequence),
            expected_actor_sequences[name],
        )

    resources = {item.resource: item for item in loop.resource_sequences}
    if len(resources) != len(loop.resource_sequences):
        raise _template_error(family, "resource sequences must be unique")
    _require_equal(
        family,
        "resource sequence set",
        set(resources),
        set(expected_resource_sequences),
    )
    for resource, expected in expected_resource_sequences.items():
        item = resources[resource]
        _require_equal(
            family,
            "%s resource sequence" % resource,
            _occurrences(item.sequence),
            expected,
        )
        if not item.source.startswith("actor_"):
            raise _template_error(
                family,
                "%s resource order must originate from actor.sequence" % resource,
            )
    return _phase_map(family, loop)


def _validate_phase_work(
    family: str,
    phases: Mapping[str, Any],
    expected: Mapping[str, Tuple[str, float, float]],
) -> None:
    _require_equal(family, "phase set", set(phases), set(expected))
    for name, (kind, flops, bytes_) in expected.items():
        phase = phases[name]
        _require_equal(family, "phase %s work kind" % name, phase.work.kind, kind)
        _require_equal(family, "phase %s FLOPs" % name, phase.work.flops, flops)
        _require_equal(family, "phase %s bytes" % name, phase.work.bytes, bytes_)


def _validate_phase_resources(
    family: str,
    phases: Mapping[str, Any],
    expected: Mapping[str, Tuple[str, ...]],
) -> None:
    _require_equal(family, "timed phase set", set(phases), set(expected))
    for name, resources in expected.items():
        timing = phases[name].timing
        if timing is None:
            raise _template_error(family, "phase %s requires static timing" % name)
        _require_equal(
            family,
            "phase %s timing resources" % name,
            tuple(item.resource for item in timing.resources),
            resources,
        )


def _validate_dependencies_and_carries(
    family: str,
    loop: PeriodicLoopIR,
    expected_dependencies: Any,
    expected_carries: Any,
) -> None:
    dependencies = {
        (
            item.source.phase_name,
            item.source.kind,
            item.target.phase_name,
            item.target.kind,
            item.iteration_distance,
            item.lag,
        )
        for item in loop.dependencies
    }
    carries = {
        (
            item.state.name,
            item.source.phase_name,
            item.source.kind,
            item.target.phase_name,
            item.target.kind,
            item.iteration_distance,
            item.lag,
        )
        for item in loop.carries
    }
    _require_equal(family, "dependency set", dependencies, set(expected_dependencies))
    _require_equal(family, "state-carry set", carries, set(expected_carries))


def _validate_pipeline_buffers(
    family: str, loop: PeriodicLoopIR, expected: Mapping[str, Tuple[Any, ...]]
) -> None:
    actual = {}
    for item in loop.pipeline_buffers:
        if item.buffer.name in actual:
            raise _template_error(family, "pipeline-buffer names must be unique")
        actual[item.buffer.name] = (
            item.capacity,
            item.acquire.phase_name,
            item.acquire.kind,
            item.release.phase_name,
            item.release.kind,
            item.minimum_residence,
        )
    _require_equal(family, "pipeline-buffer map", actual, dict(expected))


def _validate_states(
    family: str,
    loop: PeriodicLoopIR,
    expected: Mapping[str, str],
) -> None:
    actual = {}
    for item in loop.states:
        if item.name in actual:
            raise _template_error(family, "state names must be unique")
        actual[item.name] = None if item.storage is None else item.storage.name
    _require_equal(family, "state storage map", actual, dict(expected))


def _validate_buffers_and_flow(
    family: str,
    loop: PeriodicLoopIR,
    phases: Mapping[str, Any],
    expected_buffers: Mapping[str, Tuple[Any, ...]],
    expected_flow: Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]],
) -> None:
    buffers = {item.name: item for item in loop.buffers}
    if len(buffers) != len(loop.buffers):
        raise _template_error(family, "buffer names must be unique")
    _require_equal(family, "buffer set", set(buffers), set(expected_buffers))
    for name, expected in expected_buffers.items():
        item = buffers[name]
        actual = (
            item.scope,
            item.shape,
            item.dtype,
            item.slots,
            item.execution_scope,
            item.alias_group,
        )
        _require_equal(family, "buffer %s specification" % name, actual, expected)
        _require_equal(
            family, "buffer %s ownership" % name, item.ownership, None
        )

    _require_equal(family, "phase-flow set", set(phases), set(expected_flow))
    for name, (reads, writes) in expected_flow.items():
        _require_equal(
            family,
            "phase %s reads" % name,
            tuple(item.name for item in phases[name].reads),
            reads,
        )
        _require_equal(
            family,
            "phase %s writes" % name,
            tuple(item.name for item in phases[name].writes),
            writes,
        )


def _validate_lifetimes(
    family: str, loop: PeriodicLoopIR, expected: Any
) -> None:
    actual = {
        (
            item.buffer.name,
            item.acquire.phase_name,
            item.acquire.kind,
            item.release.phase_name,
            item.release.kind,
            item.minimum_residence,
        )
        for item in loop.lifetimes
    }
    _require_equal(family, "buffer lifetime set", actual, set(expected))


def _validate_frontend_template(
    kernel: KernelIR,
    spec: FlashAttentionSpec,
    scheduler_model: str,
) -> _ValidatedFATemplate:
    """Prove that the IR is the standalone source template delegated below."""

    family = spec.family
    if len(kernel.launches) != 1:
        raise _template_error(family, "exactly one launch is required")
    launch = kernel.launches[0]
    if len(launch.periodic_loops) != 1:
        raise _template_error(family, "exactly one periodic loop is required")
    loop = launch.periodic_loops[0]
    _require_equal(family, "periodic loop name", loop.name, "kv")
    _require_equal(family, "launch cluster", launch.cluster, (1, 1, 1))
    _require_equal(family, "launch residency", launch.residency, "auto")

    params = dict(kernel.params)
    if "tile" not in params:
        raise _template_error(family, "params['tile'] is required")
    tile_m, tile_n = _positive_tuple(params["tile"], "FA tile", 2)
    input_dtype = {
        "bfloat16": "bf16",
        "float16": "fp16",
    }.get(params.get("dtype", "bf16"), params.get("dtype", "bf16"))
    tile_flops = float(2 * tile_m * tile_n * spec.head_dim)
    tile_bytes = float(tile_n * spec.head_dim * spec.dtype_bytes)

    if family == "fa3":
        _require_equal(family, "launch threads", launch.threads, 256)
        expected_scheduler = (
            "persistent" if scheduler_model == "source" else "aggregate_lb"
        )
        _require_equal(
            family, "launch scheduler", launch.scheduler, expected_scheduler
        )
        _require_equal(family, "periodic stages", loop.stages, 2)
        q_stages = 1
        kv_stages = 2
        mma_pv_is_rs = not (spec.head_dim == 64 and not spec.causal)
        actor_phases = {
            "producer": ("load_K", "load_V"),
            "tensor": ("gemm_QK", "gemm_PV"),
            "softmax": ("online_softmax", "rescale_O"),
        }
        actor_sequences = {
            "producer": (("load_K", 1), ("load_V", 0)),
            "tensor": (("gemm_QK", 1), ("gemm_PV", 0)),
            "softmax": (("online_softmax", 1), ("rescale_O", 1)),
        }
        resource_sequences = {
            "tma": actor_sequences["producer"],
            "tensor": actor_sequences["tensor"],
            "cuda": actor_sequences["softmax"],
        }
        work = {
            "load_K": ("copy", 0.0, tile_bytes),
            "load_V": ("copy", 0.0, tile_bytes),
            "gemm_QK": ("mma", tile_flops, 0.0),
            "gemm_PV": ("mma", tile_flops, 0.0),
            "online_softmax": ("reduce", float(7 * tile_m * tile_n), 0.0),
            "rescale_O": ("pointwise", float(tile_m * spec.head_dim), 0.0),
        }
        phase_resources = {
            "load_K": ("tma",),
            "load_V": ("tma",),
            "gemm_QK": ("tensor",),
            "gemm_PV": ("tensor",),
            "online_softmax": (
                ("cuda", "sfu")
                if mma_pv_is_rs
                else ("smem", "cuda", "sfu")
            ),
            "rescale_O": ("cuda",),
        }
        buffers = {
            "K_shared": ("smem", (tile_n, spec.head_dim), input_dtype, 2, "cta", None),
            "V_shared": ("smem", (tile_n, spec.head_dim), input_dtype, 2, "cta", None),
            "scores": ("fragment", (tile_m, tile_n), "fp32", 1, "warpgroup", None),
            "probabilities": (
                "fragment" if mma_pv_is_rs else "smem",
                (tile_m, tile_n),
                input_dtype,
                1,
                "warpgroup" if mma_pv_is_rs else "cta",
                None,
            ),
            "O_accumulator": (
                "fragment",
                (tile_m, spec.head_dim),
                "fp32",
                1,
                "warpgroup",
                None,
            ),
            "softmax_state": (
                "fragment",
                (tile_m, 2),
                "fp32",
                1,
                "warpgroup",
                None,
            ),
        }
        flow = {
            "load_K": (tuple(), ("K_shared",)),
            "load_V": (tuple(), ("V_shared",)),
            "gemm_QK": (("K_shared",), ("scores",)),
            "gemm_PV": (
                ("probabilities", "V_shared", "O_accumulator"),
                tuple(),
            ),
            "online_softmax": (("scores",), ("probabilities",)),
            "rescale_O": (tuple(), ("O_accumulator",)),
        }
        dependencies = {
            ("online_softmax", "done", "rescale_O", "start", 0, 0.0),
            ("rescale_O", "done", "gemm_PV", "start", 0, 0.0),
        }
        carries = {
            (
                "online_softmax_state",
                "online_softmax",
                "done",
                "online_softmax",
                "start",
                1,
                0.0,
            ),
            (
                "O_accumulator",
                "gemm_PV",
                "done",
                "rescale_O",
                "start",
                1,
                0.0,
            ),
        }
        pipeline_buffers = {
            "K_shared": (2, "load_K", "start", "gemm_QK", "done", 0.0),
            "V_shared": (2, "load_V", "start", "gemm_PV", "done", 0.0),
        }
        states = {
            "online_softmax_state": "softmax_state",
            "O_accumulator": "O_accumulator",
        }
        lifetimes = {
            ("K_shared", "load_K", "start", "gemm_QK", "done", 0.0),
            ("V_shared", "load_V", "start", "gemm_PV", "done", 0.0),
        }
    else:
        _require_equal(family, "launch threads", launch.threads, 384)
        _require_equal(family, "launch scheduler", launch.scheduler, scheduler_model)
        q_stages = _positive_int(params.get("q_stages"), "FA4 q_stages")
        if q_stages not in (1, 2):
            raise _template_error(family, "q_stages must be 1 or 2")
        kv_stages = _fa4_kv_stages(
            tile_m, tile_n, spec.head_dim, q_stages, spec.dtype_bytes
        )
        _require_equal(family, "params['kv_slots']", params.get("kv_slots"), kv_stages)
        _require_equal(family, "periodic stages", loop.stages, kv_stages)
        qk = tuple("gemm_QK_s%d" % stage for stage in range(q_stages))
        pv = tuple("gemm_PV_s%d" % stage for stage in range(q_stages))
        softmax = tuple("online_softmax_s%d" % stage for stage in range(q_stages))
        correction = tuple("corr_s%d" % stage for stage in range(q_stages))
        tensor_sequence = tuple(
            item
            for stage in range(q_stages)
            for item in ((pv[stage], 0), (qk[stage], 1))
        )
        actor_phases = {
            "producer": ("load_K", "load_V"),
            "tensor": qk + pv,
            "softmax_workers": softmax,
            "correction": correction,
        }
        actor_sequences = {
            "producer": (("load_K", 0), ("load_V", 0)),
            "tensor": tensor_sequence,
            "softmax_workers": tuple(),
            "correction": tuple((name, 0) for name in correction),
        }
        resource_sequences = {
            "tma": actor_sequences["producer"],
            "tensor": actor_sequences["tensor"],
        }
        work = {
            "load_K": ("copy", 0.0, tile_bytes),
            "load_V": ("copy", 0.0, tile_bytes),
        }
        phase_resources = {
            "load_K": ("tma",),
            "load_V": ("tma",),
        }
        buffers = {
            "K_shared": (
                "smem",
                (tile_n, spec.head_dim),
                input_dtype,
                kv_stages,
                "cta",
                None,
            ),
            "V_shared": (
                "smem",
                (tile_n, spec.head_dim),
                input_dtype,
                kv_stages,
                "cta",
                None,
            ),
        }
        flow = {
            "load_K": (tuple(), ("K_shared",)),
            "load_V": (tuple(), ("V_shared",)),
        }
        dependencies = set()
        carries = set()
        pipeline_buffers = {
            "K_shared": (
                kv_stages,
                "load_K",
                "start",
                qk[-1],
                "done",
                0.0,
            ),
            "V_shared": (
                kv_stages,
                "load_V",
                "start",
                pv[-1],
                "done",
                0.0,
            ),
        }
        states = {}
        lifetimes = {
            ("K_shared", "load_K", "start", qk[-1], "done", 0.0),
            ("V_shared", "load_V", "start", pv[-1], "done", 0.0),
        }
        for stage in range(q_stages):
            work[qk[stage]] = ("mma", tile_flops, 0.0)
            work[pv[stage]] = ("mma", tile_flops, 0.0)
            work[softmax[stage]] = (
                "reduce",
                float(7 * tile_m * tile_n),
                0.0,
            )
            work[correction[stage]] = (
                "pointwise",
                float(int(tile_m * spec.head_dim * spec.correction_frequency)),
                0.0,
            )
            phase_resources[qk[stage]] = ("tensor",)
            phase_resources[pv[stage]] = ("tensor",)
            phase_resources[softmax[stage]] = ("cuda", "sfu")
            phase_resources[correction[stage]] = ("cuda",)
            alias = "sp_s%d" % stage
            buffers["S%d" % stage] = (
                "tmem",
                (tile_m, tile_n),
                "fp32",
                1,
                "cta",
                alias,
            )
            buffers["P%d" % stage] = (
                "tmem",
                (tile_m, tile_n),
                input_dtype,
                1,
                "cta",
                alias,
            )
            buffers["softmax_state_s%d" % stage] = (
                "fragment",
                (tile_m, 2),
                "fp32",
                1,
                "warpgroup",
                None,
            )
            buffers["O_accumulator_s%d" % stage] = (
                "tmem",
                (tile_m, spec.head_dim),
                "fp32",
                1,
                "cta",
                None,
            )
            flow[qk[stage]] = (("K_shared",), ("S%d" % stage,))
            flow[pv[stage]] = (("P%d" % stage, "V_shared"), tuple())
            flow[softmax[stage]] = (("S%d" % stage,), ("P%d" % stage,))
            flow[correction[stage]] = (tuple(), tuple())
            dependencies.update(
                {
                    (softmax[stage], "done", correction[stage], "start", 0, 0.0),
                    (correction[stage], "done", pv[stage], "start", 0, 0.0),
                }
            )
            carries.update(
                {
                    (
                        "online_softmax_s%d" % stage,
                        softmax[stage],
                        "done",
                        softmax[stage],
                        "start",
                        1,
                        0.0,
                    ),
                    (
                        "O_accumulator_s%d" % stage,
                        pv[stage],
                        "done",
                        correction[stage],
                        "start",
                        1,
                        0.0,
                    ),
                }
            )
            states["online_softmax_s%d" % stage] = "softmax_state_s%d" % stage
            states["O_accumulator_s%d" % stage] = "O_accumulator_s%d" % stage
            pipeline_buffers["S%d" % stage] = (
                1,
                qk[stage],
                "start",
                pv[stage],
                "done",
                0.0,
            )
            lifetimes.update(
                {
                    (
                        "S%d" % stage,
                        qk[stage],
                        "start",
                        pv[stage],
                        "done",
                        0.0,
                    ),
                    (
                        "P%d" % stage,
                        softmax[stage],
                        "start",
                        pv[stage],
                        "done",
                        0.0,
                    ),
                }
            )

    phases = _validate_actor_and_resource_sequences(
        family,
        loop,
        actor_phases,
        actor_sequences,
        resource_sequences,
    )
    _validate_phase_work(family, phases, work)
    _validate_phase_resources(family, phases, phase_resources)
    _validate_buffers_and_flow(family, loop, phases, buffers, flow)
    _validate_dependencies_and_carries(family, loop, dependencies, carries)
    _validate_states(family, loop, states)
    _validate_pipeline_buffers(family, loop, pipeline_buffers)
    _validate_lifetimes(family, loop, lifetimes)
    return _ValidatedFATemplate(
        launch=launch,
        loop=loop,
        tile_m=tile_m,
        tile_n=tile_n,
        q_stages=q_stages,
        kv_stages=kv_stages,
    )


def _validate_legacy_geometry(
    validated: _ValidatedFATemplate,
    spec: FlashAttentionSpec,
    breakdown: Mapping[str, Any],
    arch: Any,
    scheduler_model: str,
) -> None:
    family = spec.family
    launch = validated.launch
    loop = validated.loop
    expected_tile = (int(breakdown["tile_m"]), int(breakdown["tile_n"]))
    _require_equal(
        family,
        "params['tile']",
        (validated.tile_m, validated.tile_n),
        expected_tile,
    )
    _require_equal(
        family, "legacy scheduler_model", breakdown["scheduler_model"], scheduler_model
    )

    expected_work_grid = (
        int(breakdown["nq"]),
        int(breakdown["scheduled_heads"]),
        spec.batch,
    )
    _require_equal(family, "launch work_grid", launch.work_grid, expected_work_grid)
    if family == "fa3":
        expected_iterations = max(int(value) for value in breakdown["kv_iters_by_qblock"])
        valid_physical_grids = {
            ("arch.sm_count", 1, 1),
            (int(breakdown["launched_persistent_ctas"]), 1, 1),
        }
        if launch.physical_grid not in valid_physical_grids:
            raise _template_error(
                family,
                "launch physical_grid must resolve to %d persistent CTAs, got %r"
                % (breakdown["launched_persistent_ctas"], launch.physical_grid),
            )
    else:
        expected_iterations = int(breakdown["max_tile_iters"])
        _require_equal(
            family, "launch physical_grid", launch.physical_grid, expected_work_grid
        )
        _require_equal(
            family, "params['q_stages']", validated.q_stages, int(breakdown["q_stage"])
        )
    _require_equal(family, "periodic loop iterations", loop.iterations, expected_iterations)

    work_units = math.prod(expected_work_grid)
    legacy_work_units = int(
        breakdown["grid_work_tiles"] if family == "fa3" else breakdown["grid"]
    )
    _require_equal(family, "legacy grid work units", work_units, legacy_work_units)


def _load_legacy_module(family: str) -> ModuleType:
    module_name = _LEGACY_MODULE[family]
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        # Only rewrite absence of the attention backend itself. A missing
        # dependency inside an otherwise present model module is a real
        # environment error and should retain its original traceback.
        if error.name not in ("tilesight.modeling._attention", module_name):
            raise
        raise LegacyFlashAttentionModelUnavailable(
            "the audited %s whole-kernel model backend is missing; reinstall TileSight"
            % family.upper()
        ) from error


def run_legacy_flash_attention(
    kernel: KernelIR,
    arch: Any,
    *,
    inner_schedule_model: str = "resource_ii",
    scheduler_model: Optional[str] = None,
    periodic_search_profile: str = "reference",
    launch_overhead_us: float = 2.0,
    dispatch_overhead_us: float = 0.0,
) -> Tuple[FlashAttentionSpec, float, Dict[str, Any], ModuleType]:
    """Call the audited model with normalized frontend parameters verbatim.

    The returned legacy dictionary is not rewritten.  This is deliberate: it
    permits exact field-for-field parity tests and gives the common result
    converter a lossless provenance payload.
    """

    spec = resolve_flash_attention_spec(kernel)
    module = _load_legacy_module(spec.family)
    if spec.family == "fa3":
        if launch_overhead_us != 2.0:
            raise ModelingValidationError(
                "the audited FA3 model fixes kernel launch overhead at exactly 2 us"
            )
        kwargs = {
            "kv_h": spec.kv_heads,
            "causal": spec.causal,
            "arch": arch,
            "dtype_bytes": spec.dtype_bytes,
            "accum_bytes": spec.accumulator_bytes,
            "max_util": spec.max_util,
            "host_dispatch_overhead_us": dispatch_overhead_us,
            "scheduler_model": scheduler_model or "source",
            "inner_schedule_model": inner_schedule_model,
        }
        total_us, breakdown = module.model_fa3_us(
            spec.batch,
            spec.query_heads,
            spec.sequence,
            spec.head_dim,
            **kwargs
        )
    else:
        kwargs = {
            "kv_h": spec.kv_heads,
            "causal": spec.causal,
            "arch": arch,
            "dtype": spec.dtype_bytes,
            "accum": spec.accumulator_bytes,
            "max_util": spec.max_util,
            "correction_freq": spec.correction_frequency,
            "dispatch_overhead_us": dispatch_overhead_us,
            "kernel_launch_overhead_us": launch_overhead_us,
            "scheduler_model": scheduler_model or "aggregate_lb",
            "inner_schedule_model": inner_schedule_model,
            "periodic_search_profile": periodic_search_profile,
        }
        total_us, breakdown = module.model_fa4_us(
            spec.batch,
            spec.query_heads,
            spec.sequence,
            spec.head_dim,
            **kwargs
        )

    return spec, total_us, breakdown, module


def legacy_component_times_s(
    spec: FlashAttentionSpec,
    breakdown: Mapping[str, Any],
    module: ModuleType,
    arch: Any,
) -> Dict[str, float]:
    """Expose prologue/steady/epilogue/launch/dispatch in SI seconds.

    FA3 already publishes the three body components.  FA4's legacy dictionary
    publishes steady time plus a combined per-tile overhead, so the adapter
    evaluates the same private epilogue and Q-load primitives to split that
    overhead without changing the legacy model or its returned total.
    """

    if spec.family == "fa3":
        return {
            "prologue": float(breakdown["t_prologue_us"]) * 1.0e-6,
            "steady": float(breakdown["t_steady_us"]) * 1.0e-6,
            "epilogue": float(breakdown["t_epilogue_us"]) * 1.0e-6,
            "launch": float(breakdown["kernel_launch_us"]) * 1.0e-6,
            "dispatch": float(breakdown["host_dispatch_us"]) * 1.0e-6,
        }

    tile_m = int(breakdown["tile_m"])
    q_stage = int(breakdown["q_stage"])
    query_tile_m = int(breakdown["QM"])
    busy_tiles = float(breakdown["tiles_busy"])
    q_load_s = module.make_op_group(
        "load_Q",
        arch,
        ddr_io=query_tile_m * spec.head_dim * spec.dtype_bytes,
        l2_io=query_tile_m * spec.head_dim * spec.dtype_bytes,
        smem_io=query_tile_m * spec.head_dim * spec.dtype_bytes,
        max_util=spec.max_util,
    ).latency
    epilogue_s = module._epilogue_latency(
        arch,
        tile_m,
        spec.head_dim,
        q_stage,
        spec.dtype_bytes,
        spec.accumulator_bytes,
        spec.max_util,
    )
    serial_fill_s = float(breakdown["iter_lat_serial_ns"]) * 1.0e-9
    return {
        "prologue": busy_tiles * (q_load_s + serial_fill_s),
        "steady": float(breakdown["t_steady_us"]) * 1.0e-6,
        "epilogue": busy_tiles * epilogue_s,
        "launch": float(breakdown["kernel_launch_us"]) * 1.0e-6,
        "dispatch": float(breakdown["dispatch_us"]) * 1.0e-6,
    }


def legacy_grid_metadata(
    spec: FlashAttentionSpec, breakdown: Mapping[str, Any], arch: Any
) -> Dict[str, Any]:
    """Normalize work-grid, source scheduler, wave, and critical-path fields."""

    if spec.family == "fa3":
        return {
            "work_tiles": int(breakdown["grid_work_tiles"]),
            "physical_ctas": int(breakdown["launched_persistent_ctas"]),
            "waves": float(breakdown["waves"]),
            "critical_tiles": breakdown["critical_tiles"],
            "critical_iterations": breakdown["critical_kv_iters"],
            "scheduler": breakdown["scheduler"],
            "scheduler_model": breakdown["scheduler_model"],
            "wave_tail_mode": "persistent_makespan",
        }
    return {
        "work_tiles": int(breakdown["grid"]),
        "physical_ctas": int(breakdown["grid"]),
        "waves": float(breakdown["waves"]),
        "critical_tiles": breakdown["tiles_busy"],
        "critical_iterations": breakdown["busy_kv_iters"],
        "scheduler": breakdown["scheduler"],
        "scheduler_model": breakdown["scheduler_model"],
        "wave_tail_mode": "nonpersistent_wave_tail",
    }


def legacy_occupancy_metadata(
    spec: FlashAttentionSpec, breakdown: Mapping[str, Any], arch: Any
) -> Dict[str, Any]:
    """Describe the spatial concurrency actually used by the audited models."""

    work_tiles = (
        int(breakdown["grid_work_tiles"])
        if spec.family == "fa3"
        else int(breakdown["grid"])
    )
    # Both audited FA paths intentionally model one resident CTA stream per SM.
    # This is source behavior, not a footprint-derived generic occupancy claim.
    return {
        "resident_ctas_per_sm": 1,
        "active_sms": min(int(arch.sm_count), work_tiles),
        "sm_count": int(arch.sm_count),
        "scope": "source_fixed_one_cta_per_sm",
    }


def legacy_resource_metadata(
    spec: FlashAttentionSpec, breakdown: Mapping[str, Any]
) -> Dict[str, Any]:
    """Normalize II selection and retain every constructive search diagnostic."""

    result = {
        "ii_mode": breakdown["inner_schedule_model"],
        "ii_model": breakdown["ii_model"],
        "selected_ii_s": float(breakdown["ii_selected_ns"]) * 1.0e-9,
        "resource_ii_lower_bound_s": float(breakdown["resource_ii_lb_ns"])
        * 1.0e-9,
        "selection_label": breakdown["ii_selection_label"],
        "selection_scope": breakdown["ii_selection_scope"],
        "is_constructive": breakdown["ii_is_constructive"],
        "search_complete": breakdown["ii_search_complete"],
    }
    if spec.family == "fa3":
        result["search"] = dict(breakdown["ii_metadata"])
    else:
        periodic_keys = (
            "periodic_search_profile",
            "periodic_topology_trials",
            "kv_stage",
            "periodic_best_ii_ns",
            "periodic_worst_ii_ns",
            "periodic_spread_ns",
            "periodic_spread_pct",
            "periodic_orderings_explored",
            "periodic_infeasible_orderings",
            "periodic_unsupported_orderings",
            "periodic_search_complete",
            "periodic_endpoint_scope",
            "periodic_best_resource_orders",
            "periodic_worst_resource_orders",
        )
        result["search"] = {
            key: breakdown[key] for key in periodic_keys if key in breakdown
        }
    return result


def legacy_provenance(
    spec: FlashAttentionSpec, module: ModuleType
) -> Dict[str, Any]:
    commit_name = (
        "AUDITED_FLASH_ATTN_SOURCE_COMMIT"
        if spec.family == "fa3"
        else "FLASH_ATTN_SOURCE_COMMIT"
    )
    return {
        "adapter": "legacy_flash_attention_source_tree",
        "model_family": spec.family,
        "legacy_module": module.__name__,
        "source_commit": getattr(module, commit_name),
        "units": "SI_seconds",
    }


def _scheduler_model(family: str, grid_policy: str) -> str:
    choices = {
        "fa3": {
            "model_default": "source",
            "source": "source",
            "aggregate_lb": "aggregate_lb",
        },
        "fa4": {
            "model_default": "aggregate_lb",
            "aggregate_lb": "aggregate_lb",
            "sectioned_lpt": "sectioned_lpt",
        },
    }
    try:
        return choices[family][grid_policy]
    except KeyError:
        raise ModelingValidationError(
            "%s grid_policy must be one of %s; got %r"
            % (family.upper(), sorted(choices[family]), grid_policy)
        ) from None


def _ii_summary(
    spec: FlashAttentionSpec, breakdown: Mapping[str, Any]
) -> IISummary:
    if spec.family == "fa3":
        search = breakdown["ii_metadata"]
        best_ns = search.get("periodic_best_ii_ns")
        worst_ns = search.get("periodic_worst_ii_ns")
    else:
        best_ns = breakdown.get("periodic_best_ii_ns")
        worst_ns = breakdown.get("periodic_worst_ii_ns")
    return IISummary(
        selected_s=float(breakdown["ii_selected_ns"]) * 1.0e-9,
        resource_lower_bound_s=float(breakdown["resource_ii_lb_ns"])
        * 1.0e-9,
        best_s=None if best_ns is None else float(best_ns) * 1.0e-9,
        worst_s=None if worst_ns is None else float(worst_ns) * 1.0e-9,
        scope=str(breakdown["ii_selection_scope"]),
        constructive=breakdown["ii_is_constructive"],
        search_complete=breakdown["ii_search_complete"],
    )


def _timing_result(
    spec: FlashAttentionSpec,
    breakdown: Mapping[str, Any],
    total_us: float,
    components: Mapping[str, float],
) -> TimingBreakdown:
    if spec.family == "fa3":
        kernel_body_us = breakdown["gpu_body_us"]
        launch_us = breakdown["kernel_launch_us"]
        dispatch_us = breakdown["host_dispatch_us"]
    else:
        kernel_body_us = breakdown["kernel_body_us"]
        launch_us = breakdown["kernel_launch_us"]
        dispatch_us = breakdown["dispatch_us"]
    # Every parity-bearing top-level value comes directly from the legacy
    # result.  Components are diagnostic and are not used to reconstruct it.
    return TimingBreakdown(
        kernel_body_s=float(kernel_body_us) * 1.0e-6,
        launch_s=float(launch_us) * 1.0e-6,
        host_dispatch_s=float(dispatch_us) * 1.0e-6,
        total_s=float(total_us) * 1.0e-6,
        prologue_s=components["prologue"],
        steady_s=components["steady"],
        epilogue_s=components["epilogue"],
        component_scope="critical_sm_path",
    )


def _grid_result(
    spec: FlashAttentionSpec, breakdown: Mapping[str, Any], arch: Any
) -> GridResult:
    if spec.family == "fa3":
        work_grid = (
            int(breakdown["nq"]),
            int(breakdown["scheduled_heads"]),
            spec.batch,
        )
        physical_grid = (int(breakdown["launched_persistent_ctas"]), 1, 1)
        total = float(breakdown["grid_work_tiles"])
        critical = float(breakdown["critical_tiles"])
    else:
        work_grid = (
            int(breakdown["nq"]),
            int(breakdown["scheduled_heads"]),
            spec.batch,
        )
        physical_grid = work_grid
        total = float(breakdown["grid"])
        critical = float(breakdown["tiles_busy"])
    return GridResult(
        work_grid=work_grid,
        physical_grid=physical_grid,
        scheduler=str(breakdown["scheduler"]),
        total_work_units=total,
        waves=float(breakdown["waves"]),
        # The persistent FA3 makespan and FA4 weighted aggregate have no
        # single occupancy-ratio tail scalar that is exact for every q-block.
        tail_factor=None,
        critical_work_units=critical,
    )


def _occupancy_result(kernel: KernelIR) -> OccupancyResult:
    launch = kernel.launches[0]
    cluster_size = 1
    for extent in launch.cluster:
        cluster_size *= extent
    # These audited FA models execute one CTA stream per SM.  Liveness exposes
    # the static storage envelope separately; it is not used to invent a new
    # occupancy formula and thereby change parity.
    return OccupancyResult(
        resident_ctas_per_sm=1.0,
        warps_per_cta=launch.threads / 32.0,
        cluster_size=cluster_size,
    )


def _model_family(
    expected_family: str,
    kernel: KernelIR,
    arch: Any,
    options: FullModelOptions,
) -> FullModelResult:
    if not isinstance(options, FullModelOptions):
        raise ModelingValidationError("FA full model expects FullModelOptions")
    if kernel.fusion_plans:
        raise FullModelError(
            "%s parity adapter models a standalone legacy kernel and cannot "
            "return fused latency for a nonempty FusionPlan; use analyze_fusion() "
            "for traffic/liveness until a fused executor is selected"
            % expected_family.upper()
        )
    spec = resolve_flash_attention_spec(kernel)
    if spec.family != expected_family:
        raise ModelingValidationError(
            "%s adapter cannot model family %r" % (expected_family.upper(), spec.family)
        )
    adapter_options = dict(options.adapter_options)
    if adapter_options:
        raise ModelingValidationError(
            "unsupported %s adapter_options: %s"
            % (expected_family.upper(), ", ".join(sorted(adapter_options)))
        )

    launch_us = (
        2.0 if options.kernel_launch_s is None else options.kernel_launch_s * 1.0e6
    )
    scheduler_model = _scheduler_model(spec.family, options.grid_policy)
    validated = _validate_frontend_template(kernel, spec, scheduler_model)
    spec, total_us, breakdown, module = run_legacy_flash_attention(
        kernel,
        arch,
        inner_schedule_model=options.ii_mode,
        scheduler_model=scheduler_model,
        periodic_search_profile=options.search_profile,
        launch_overhead_us=launch_us,
        dispatch_overhead_us=options.host_dispatch_s * 1.0e6,
    )
    _validate_legacy_geometry(
        validated, spec, breakdown, arch, scheduler_model
    )
    components = legacy_component_times_s(spec, breakdown, module, arch)
    resources = legacy_resource_metadata(spec, breakdown)
    resources["grid"] = legacy_grid_metadata(spec, breakdown, arch)
    resources["occupancy"] = legacy_occupancy_metadata(spec, breakdown, arch)

    return FullModelResult(
        adapter=spec.family,
        timing=_timing_result(spec, breakdown, total_us, components),
        ii=_ii_summary(spec, breakdown),
        occupancy=_occupancy_result(kernel),
        grid=_grid_result(spec, breakdown, arch),
        resource_metrics=resources,
        liveness=analyze_liveness(
            kernel,
            arch=arch,
            resident_ctas_per_sm=1.0,
        ),
        fusion=analyze_fusion(kernel),
        legacy=breakdown,
        provenance=legacy_provenance(spec, module),
    )


def model_fa3(
    kernel: KernelIR, arch: Any, options: FullModelOptions
) -> FullModelResult:
    """Return exact whole-kernel parity with ``model_fa3_us``."""

    return _model_family("fa3", kernel, arch, options)


def model_fa4(
    kernel: KernelIR, arch: Any, options: FullModelOptions
) -> FullModelResult:
    """Return exact whole-kernel parity with ``model_fa4_us``."""

    return _model_family("fa4", kernel, arch, options)


__all__ = [
    "FlashAttentionSpec",
    "LegacyFlashAttentionModelUnavailable",
    "legacy_component_times_s",
    "legacy_grid_metadata",
    "legacy_occupancy_metadata",
    "legacy_provenance",
    "legacy_resource_metadata",
    "model_fa3",
    "model_fa4",
    "resolve_flash_attention_spec",
    "run_legacy_flash_attention",
]
