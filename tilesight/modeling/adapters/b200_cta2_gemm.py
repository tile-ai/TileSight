"""Source-shaped B200 two-CTA cooperative GEMM prototype.

The calibrated compatibility adapter treats a two-SM UTCMMA tile as one
logical work unit.  That is the right *spatial* unit, but by itself it loses
the fact that the work unit contains two physical CTA participants.  This
module keeps both views:

* the launch grid counts logical cooperative supertiles;
* each supertile contains rank-0/rank-1 local loads and local SMEM buffers;
* an operand is explicitly partitioned, duplicated, multicast, or read via
  peer SMEM;
* only multicast/peer-SMEM routes create payload-bearing cluster transfers;
* cluster-ready, pair issue, and ``cta_group::2`` Tensor work remain explicit
  phases in the periodic DAG.

The first cost binding intentionally reuses the legacy model's measured-bound
per-iteration memory/Tensor/store times.  It therefore audits the cooperative
schedule and transfer protocol; it is not an independent hardware calibration.
Barrier, issue, and optional link costs are nevertheless consumed explicitly,
so a caller can run sensitivity studies without inventing a generic DSM
bandwidth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Tuple

from tilesight.modeling._pipeline.periodic_schedule import SearchConfig

from ..errors import ModelingValidationError
from ..frontend import Kernel
from ..full_model import FeasibilityStatus, LivenessReport, StoragePeak
from ..ir import KernelIR, Phase, ResourceTiming, Timing, Work, freeze_attrs
from ..native_executor import NativeModelOptions, NativeModelResult, model_native
from ..native_policy import LaunchTopology
from ..ops import DTypeSpec
from ..regions import LoopRegion, PeriodicAxisIR, PhaseRegion, SequenceRegion
from .gemm import GemmLegacyEvaluation, evaluate_gemm_legacy


_ROUTE_MODES = frozenset(("partition", "duplicate", "multicast", "peer_smem"))
_TRANSFER_KINDS = frozenset(("none", "tma_multicast", "peer_smem"))
_PHYSICAL_PATHS = frozenset(("tma_local", "tma_multicast", "peer_smem"))


def _finite_nonnegative(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ModelingValidationError("%s must be numeric" % label) from error
    if not math.isfinite(result) or result < 0.0:
        raise ModelingValidationError("%s must be finite and non-negative" % label)
    return result


@dataclass(frozen=True)
class Cta2OperandRoute:
    """Physical placement of one K-step operand across two CTA ranks.

    ``local_bytes`` describes bytes resident in each CTA's local SMEM.
    ``global_load_bytes`` describes requests independently issued by each CTA.
    ``transfer_bytes`` is nonzero only when a payload is delivered from one
    participant/fabric endpoint to the other participant.
    """

    operand: str
    mode: str
    local_bytes: Tuple[float, float]
    global_load_bytes: Tuple[float, float]
    transfer_kind: str = "none"
    transfer_bytes: float = 0.0
    source_rank: int = 0
    target_rank: int = 1

    def __post_init__(self) -> None:
        if self.operand not in ("A", "B"):
            raise ModelingValidationError("CTA2 operand must be A or B")
        if self.mode not in _ROUTE_MODES:
            raise ModelingValidationError("unsupported CTA2 operand route mode")
        if self.transfer_kind not in _TRANSFER_KINDS:
            raise ModelingValidationError("unsupported CTA2 transfer kind")
        local = tuple(
            _finite_nonnegative(item, "%s local bytes" % self.operand)
            for item in self.local_bytes
        )
        loads = tuple(
            _finite_nonnegative(item, "%s global bytes" % self.operand)
            for item in self.global_load_bytes
        )
        if len(local) != 2 or len(loads) != 2 or min(local) <= 0.0:
            raise ModelingValidationError(
                "CTA2 operand routes need two positive local sizes and two loads"
            )
        transfer = _finite_nonnegative(
            self.transfer_bytes, "%s transfer bytes" % self.operand
        )
        if self.source_rank not in (0, 1) or self.target_rank not in (0, 1):
            raise ModelingValidationError("CTA2 transfer ranks must be zero or one")
        if self.source_rank == self.target_rank:
            raise ModelingValidationError("CTA2 transfer source and target must differ")
        if self.mode in ("partition", "duplicate"):
            if self.transfer_kind != "none" or transfer != 0.0:
                raise ModelingValidationError(
                    "%s route cannot carry a peer payload" % self.mode
                )
            if min(loads) <= 0.0:
                raise ModelingValidationError(
                    "%s route requires both CTAs to issue global loads" % self.mode
                )
        else:
            expected_kind = "tma_multicast" if self.mode == "multicast" else "peer_smem"
            if self.transfer_kind != expected_kind or transfer <= 0.0:
                raise ModelingValidationError(
                    "%s route needs a matching positive transfer" % self.mode
                )
            if loads[self.source_rank] <= 0.0 or loads[self.target_rank] != 0.0:
                raise ModelingValidationError(
                    "%s route must load only at its source rank" % self.mode
                )
        object.__setattr__(self, "local_bytes", local)
        object.__setattr__(self, "global_load_bytes", loads)
        object.__setattr__(self, "transfer_bytes", transfer)


@dataclass(frozen=True)
class Cta2PhysicalTransfer:
    """One source-declared physical movement into a CTA-local SMEM shard.

    A logical operand may have several transfers.  In particular, partition
    and multicast are not mutually exclusive at larger launch-cluster scopes:
    one rank-sharded TMA transaction may multicast that shard to more than one
    destination CTA.  The route summary above is convenient for the 2CTA DAG;
    this transfer list is the normalized traffic record.  The current 2CTA
    vertical slice checks it exactly against the compact route contract before
    either scheduling or pricing it.
    """

    name: str
    operand: str
    shard: str
    issuer_rank: int
    destination_ranks: Tuple[int, ...]
    path: str
    payload_bytes: float
    source_level: str = "ddr"
    target_level: str = "smem"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ModelingValidationError("physical transfer name must be nonempty")
        if self.operand not in ("A", "B"):
            raise ModelingValidationError("physical transfer operand must be A or B")
        if not isinstance(self.shard, str) or not self.shard:
            raise ModelingValidationError("physical transfer shard must be nonempty")
        if self.issuer_rank not in (0, 1):
            raise ModelingValidationError("physical transfer issuer must be rank 0/1")
        destinations = tuple(self.destination_ranks)
        if (
            not destinations
            or len(destinations) != len(set(destinations))
            or any(item not in (0, 1) for item in destinations)
        ):
            raise ModelingValidationError("invalid physical transfer destinations")
        if self.path not in _PHYSICAL_PATHS:
            raise ModelingValidationError("unsupported physical transfer path")
        if self.path == "tma_local" and destinations != (self.issuer_rank,):
            raise ModelingValidationError("tma_local must target only its issuer")
        if self.path == "tma_multicast" and len(destinations) < 2:
            raise ModelingValidationError("tma_multicast needs multiple destinations")
        if self.path == "peer_smem" and (
            len(destinations) != 1 or destinations[0] == self.issuer_rank
        ):
            raise ModelingValidationError("peer_smem must target the peer rank")
        if self.path in ("tma_local", "tma_multicast") and (
            self.source_level != "ddr" or self.target_level != "smem"
        ):
            raise ModelingValidationError(
                "TMA physical transfers must move ddr to smem"
            )
        if self.path == "peer_smem" and (
            self.source_level != "smem" or self.target_level != "smem"
        ):
            raise ModelingValidationError(
                "peer_smem physical transfers must move smem to smem"
            )
        payload = _finite_nonnegative(self.payload_bytes, "physical transfer bytes")
        if payload <= 0.0:
            raise ModelingValidationError("physical transfer bytes must be positive")
        object.__setattr__(self, "destination_ranks", destinations)
        object.__setattr__(self, "payload_bytes", payload)

    @property
    def crosses_sms(self) -> bool:
        return any(rank != self.issuer_rank for rank in self.destination_ranks)


@dataclass(frozen=True)
class B200Cta2ExecutionPlan:
    """One source-derived two-CTA operand and control protocol."""

    cluster_axis: str
    routes: Tuple[Cta2OperandRoute, ...]
    physical_transfers: Tuple[Cta2PhysicalTransfer, ...] = field(default_factory=tuple)
    leader_rank: int = 0
    source: str = "explicit_user_plan"
    pair_tensor_protocol: str = "tcgen05_mma_cta_group_2"
    load_ready_protocol: str = "tma_2sm_transaction_barrier"
    input_release_protocol: str = "umma_consumed_multicast_2x1sm"
    completion_protocol: str = "tmem_full_pair_completion"

    def __post_init__(self) -> None:
        if self.cluster_axis not in ("M", "N"):
            raise ModelingValidationError("CTA2 cluster axis must be M or N")
        routes = tuple(self.routes)
        if len(routes) != 2 or {item.operand for item in routes} != {"A", "B"}:
            raise ModelingValidationError("CTA2 plan needs exactly A and B routes")
        if self.leader_rank not in (0, 1):
            raise ModelingValidationError("CTA2 leader rank must be zero or one")
        if not isinstance(self.source, str) or not self.source:
            raise ModelingValidationError("CTA2 plan source must be nonempty")
        object.__setattr__(self, "routes", routes)
        transfers = tuple(self.physical_transfers)
        if not transfers:
            generated = []
            for route in routes:
                if route.mode in ("partition", "duplicate"):
                    for rank in (0, 1):
                        generated.append(
                            Cta2PhysicalTransfer(
                                name="%s_rank%d_tma" % (route.operand, rank),
                                operand=route.operand,
                                shard=(
                                    "rank%d_partition" % rank
                                    if route.mode == "partition"
                                    else "replica"
                                ),
                                issuer_rank=rank,
                                destination_ranks=(rank,),
                                path="tma_local",
                                payload_bytes=route.global_load_bytes[rank],
                            )
                        )
                elif route.mode == "multicast":
                    generated.append(
                        Cta2PhysicalTransfer(
                            name="%s_multicast" % route.operand,
                            operand=route.operand,
                            shard="common",
                            issuer_rank=route.source_rank,
                            destination_ranks=(route.source_rank, route.target_rank),
                            path="tma_multicast",
                            payload_bytes=route.global_load_bytes[route.source_rank],
                        )
                    )
                else:
                    generated.extend(
                        (
                            Cta2PhysicalTransfer(
                                name="%s_source_tma" % route.operand,
                                operand=route.operand,
                                shard="common",
                                issuer_rank=route.source_rank,
                                destination_ranks=(route.source_rank,),
                                path="tma_local",
                                payload_bytes=route.global_load_bytes[route.source_rank],
                            ),
                            Cta2PhysicalTransfer(
                                name="%s_peer_smem" % route.operand,
                                operand=route.operand,
                                shard="common",
                                issuer_rank=route.source_rank,
                                destination_ranks=(route.target_rank,),
                                path="peer_smem",
                                payload_bytes=route.transfer_bytes,
                                source_level="smem",
                            ),
                        )
                    )
            transfers = tuple(generated)
        if not all(isinstance(item, Cta2PhysicalTransfer) for item in transfers):
            raise ModelingValidationError(
                "physical_transfers must contain Cta2PhysicalTransfer"
            )
        names = [item.name for item in transfers]
        if len(names) != len(set(names)):
            raise ModelingValidationError("physical transfer names must be unique")
        for route in routes:
            operand_transfers = tuple(
                item for item in transfers if item.operand == route.operand
            )
            signature = tuple(
                sorted(
                    (
                        item.path,
                        item.issuer_rank,
                        item.destination_ranks,
                    )
                    for item in operand_transfers
                )
            )
            if route.mode in ("partition", "duplicate"):
                expected_signature = (
                    ("tma_local", 0, (0,)),
                    ("tma_local", 1, (1,)),
                )
            elif route.mode == "multicast":
                expected_signature = (
                    (
                        "tma_multicast",
                        route.source_rank,
                        (route.source_rank, route.target_rank),
                    ),
                )
            else:
                expected_signature = tuple(
                    sorted(
                        (
                            (
                                "tma_local",
                                route.source_rank,
                                (route.source_rank,),
                            ),
                            (
                                "peer_smem",
                                route.source_rank,
                                (route.target_rank,),
                            ),
                        )
                    )
                )
            if signature != expected_signature:
                raise ModelingValidationError(
                    "physical transfer paths for operand %s disagree with "
                    "route mode %s" % (route.operand, route.mode)
                )
            per_rank_source = tuple(
                sum(
                    item.payload_bytes
                    for item in operand_transfers
                    if item.issuer_rank == rank
                    and item.path in ("tma_local", "tma_multicast")
                )
                for rank in (0, 1)
            )
            per_rank_delivery = tuple(
                sum(
                    item.payload_bytes
                    for item in operand_transfers
                    if rank in item.destination_ranks
                )
                for rank in (0, 1)
            )
            if any(
                not math.isclose(lhs, rhs, rel_tol=1.0e-12, abs_tol=1.0e-12)
                for lhs, rhs in zip(per_rank_source, route.global_load_bytes)
            ) or any(
                not math.isclose(lhs, rhs, rel_tol=1.0e-12, abs_tol=1.0e-12)
                for lhs, rhs in zip(per_rank_delivery, route.local_bytes)
            ):
                raise ModelingValidationError(
                    "physical transfer per-rank bytes for operand %s disagree "
                    "with route placement" % route.operand
                )
            source_bytes = sum(
                item.payload_bytes
                for item in operand_transfers
                if item.path in ("tma_local", "tma_multicast")
            )
            delivered_bytes = sum(
                item.payload_bytes * len(item.destination_ranks)
                for item in operand_transfers
            )
            peer_bytes = sum(
                item.payload_bytes for item in operand_transfers if item.crosses_sms
            )
            expected = (
                sum(route.global_load_bytes),
                sum(route.local_bytes),
                route.transfer_bytes,
            )
            actual = (source_bytes, delivered_bytes, peer_bytes)
            if any(
                not math.isclose(lhs, rhs, rel_tol=1.0e-12, abs_tol=1.0e-12)
                for lhs, rhs in zip(actual, expected)
            ):
                raise ModelingValidationError(
                    "physical transfers for operand %s disagree with route bytes "
                    "(source, delivered, cross-SM)=%r expected=%r"
                    % (route.operand, actual, expected)
                )
        object.__setattr__(self, "physical_transfers", transfers)

    @property
    def route_map(self) -> Mapping[str, Cta2OperandRoute]:
        return {item.operand: item for item in self.routes}

    @property
    def payload_transfer_bytes_per_iteration(self) -> float:
        return sum(
            item.payload_bytes for item in self.physical_transfers if item.crosses_sms
        )

    @property
    def global_request_bytes_per_iteration(self) -> float:
        return sum(
            item.payload_bytes
            for item in self.physical_transfers
            if item.path in ("tma_local", "tma_multicast")
        )

    @property
    def delivered_smem_bytes_per_iteration(self) -> float:
        return sum(
            item.payload_bytes * len(item.destination_ranks)
            for item in self.physical_transfers
        )

    @property
    def has_tma_multicast(self) -> bool:
        return any(item.transfer_kind == "tma_multicast" for item in self.routes)

    @property
    def has_peer_smem(self) -> bool:
        return any(item.transfer_kind == "peer_smem" for item in self.routes)

    @property
    def topology_operand_plan(self) -> Tuple[Tuple[str, str], ...]:
        return tuple((item.operand, item.mode) for item in self.routes)

    @classmethod
    def from_legacy_spec(
        cls,
        evaluation: GemmLegacyEvaluation,
        *,
        plan_kind: str,
        source: str,
    ) -> "B200Cta2ExecutionPlan":
        """Build the physical routes for the compatibility tile geometry.

        The source-backed SM100 dense SS plan partitions both operands: each
        CTA owns one M half of A and one N half of B, while the pair instruction
        combines them to produce each CTA's local C shard.  The legacy model's
        older broadcast interpretation is retained only as an explicitly named
        sensitivity mode.
        """

        allowed_plans = (
            "source_partitioned",
            "legacy_broadcast_duplicate",
            "legacy_broadcast_multicast",
            "legacy_broadcast_peer_smem",
        )
        if plan_kind not in allowed_plans:
            raise ModelingValidationError(
                "plan_kind must be one of %s" % ", ".join(allowed_plans)
            )
        axis = evaluation.wave.cluster_axis
        if axis not in ("M", "N") or evaluation.wave.effective_mma_type != "utcmma_cta2":
            raise ModelingValidationError("evaluation is not an effective CTA2 GEMM")
        spec = evaluation.spec
        tm, tn, tk = spec.tb_shape
        levels = spec.legacy_mem_levels()
        a_bytes = float(tm * tk * levels["in1"][-1])
        b_bytes = float(tk * tn * levels["in2"][-1])

        def partition(name: str, size: float) -> Cta2OperandRoute:
            return Cta2OperandRoute(
                name,
                "partition",
                local_bytes=(size, size),
                global_load_bytes=(size, size),
            )

        def broadcast(name: str, size: float) -> Cta2OperandRoute:
            broadcast_mode = plan_kind.replace("legacy_broadcast_", "")
            if broadcast_mode == "duplicate":
                return Cta2OperandRoute(
                    name,
                    "duplicate",
                    local_bytes=(size, size),
                    global_load_bytes=(size, size),
                )
            return Cta2OperandRoute(
                name,
                broadcast_mode,
                local_bytes=(size, size),
                global_load_bytes=(size, 0.0),
                transfer_kind=(
                    "tma_multicast"
                    if broadcast_mode == "multicast"
                    else "peer_smem"
                ),
                transfer_bytes=size,
            )

        if plan_kind == "source_partitioned":
            if axis != "M":
                raise ModelingValidationError(
                    "source-backed SM100 dense CTA2 plan currently requires the "
                    "2x1 M pair used by tcgen05.mma cta_group::2"
                )
            # ``tb_m`` is the physical-CTA M tile in the compatibility input;
            # the pair logical tile is 2*tb_m.  B's logical N is shared by the
            # instruction, but each rank-local SMEM shard holds N/2.
            if tn % 2:
                raise ModelingValidationError(
                    "source-partitioned CTA2 requires an even N tile"
                )
            routes = (
                partition("A", a_bytes),
                partition("B", b_bytes / 2.0),
            )
        elif axis == "M":
            routes = (partition("A", a_bytes), broadcast("B", b_bytes))
        else:
            routes = (broadcast("A", a_bytes), partition("B", b_bytes))
        return cls(axis, routes, source=source)


@dataclass(frozen=True)
class Cta2ProtocolCosts:
    """Explicit costs not identifiable from the legacy GEMM resource bound.

    ``included_in_legacy_memory`` records transfer payload in the IR while
    leaving its latency folded into the calibrated memory phase.  Selecting
    ``explicit_link`` requires a bandwidth and/or fixed latency and creates a
    separately scheduled cluster-link reservation.

    ``tma_fixed_latency_s`` is an end-to-end completion delay added to every
    local TMA phase after its throughput-derived service time. It delays the
    ready event without occupying ``tma_rank*`` for longer, allowing the stage
    token recurrence to determine whether the latency is actually hidden.
    """

    transfer_pricing: str = "included_in_legacy_memory"
    transfer_bandwidth_bytes_per_s: Optional[float] = None
    transfer_fixed_latency_s: float = 0.0
    tma_fixed_latency_s: float = 0.0
    cluster_barrier_s: float = 0.0
    cluster_issue_s: float = 0.0
    input_consumed_latency_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.transfer_pricing not in (
            "included_in_legacy_memory",
            "explicit_link",
        ):
            raise ModelingValidationError("unsupported CTA2 transfer pricing")
        bandwidth = self.transfer_bandwidth_bytes_per_s
        if bandwidth is not None:
            bandwidth = _finite_nonnegative(bandwidth, "CTA2 transfer bandwidth")
            if bandwidth <= 0.0:
                raise ModelingValidationError("CTA2 transfer bandwidth must be positive")
            object.__setattr__(self, "transfer_bandwidth_bytes_per_s", bandwidth)
        for name in (
            "transfer_fixed_latency_s",
            "tma_fixed_latency_s",
            "cluster_barrier_s",
            "cluster_issue_s",
        ):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )
        if self.input_consumed_latency_s is not None:
            object.__setattr__(
                self,
                "input_consumed_latency_s",
                _finite_nonnegative(
                    self.input_consumed_latency_s,
                    "input_consumed_latency_s",
                ),
            )
        if (
            self.transfer_pricing == "explicit_link"
            and bandwidth is None
            and self.transfer_fixed_latency_s <= 0.0
        ):
            raise ModelingValidationError(
                "explicit CTA2 link pricing needs bandwidth or fixed latency"
            )

    def transfer_time(self, route: Cta2OperandRoute) -> float:
        if route.transfer_bytes <= 0.0 or self.transfer_pricing == "included_in_legacy_memory":
            return 0.0
        value = self.transfer_fixed_latency_s
        if self.transfer_bandwidth_bytes_per_s is not None:
            value += route.transfer_bytes / self.transfer_bandwidth_bytes_per_s
        return value


@dataclass(frozen=True)
class B200Cta2NativeOptions:
    """Options for the cooperative vertical slice."""

    plan_kind: str = "source_partitioned"
    plan_source: str = "cutlass_sm100_dense_2sm_partition_contract"
    protocol_costs: Cta2ProtocolCosts = field(default_factory=Cta2ProtocolCosts)
    ii_mode: str = "periodic_best"
    boundary_policy: str = "periodic_witness"
    wave_policy: str = "legacy_calibrated"
    kernel_launch_s: float = 2.0e-6
    host_dispatch_s: float = 0.0
    search_config: SearchConfig = field(default_factory=SearchConfig)

    def __post_init__(self) -> None:
        if self.plan_kind not in (
            "source_partitioned",
            "legacy_broadcast_duplicate",
            "legacy_broadcast_multicast",
            "legacy_broadcast_peer_smem",
        ):
            raise ModelingValidationError("unsupported CTA2 operand plan kind")
        if not isinstance(self.protocol_costs, Cta2ProtocolCosts):
            raise ModelingValidationError("protocol_costs must be Cta2ProtocolCosts")
        if self.ii_mode not in ("periodic_best", "periodic_worst"):
            raise ModelingValidationError("CTA2 native modeling needs a periodic witness")
        if self.boundary_policy not in ("finite_witness", "periodic_witness"):
            raise ModelingValidationError(
                "CTA2 boundary_policy must be finite_witness or periodic_witness"
            )
        if self.wave_policy not in ("native", "legacy_calibrated"):
            raise ModelingValidationError("unsupported CTA2 wave policy")
        if not isinstance(self.search_config, SearchConfig):
            raise ModelingValidationError("search_config must be SearchConfig")
        for name in ("kernel_launch_s", "host_dispatch_s"):
            object.__setattr__(
                self, name, _finite_nonnegative(getattr(self, name), name)
            )


@dataclass(frozen=True)
class Cta2ParticipantFootprint:
    """Static storage owned by one physical CTA/SM participant."""

    rank: int
    smem_bytes_per_stage: float
    smem_all_stages_bytes: float
    tmem_bytes: float

    def __post_init__(self) -> None:
        if self.rank not in (0, 1):
            raise ModelingValidationError("CTA2 footprint rank must be zero or one")
        for name in (
            "smem_bytes_per_stage",
            "smem_all_stages_bytes",
            "tmem_bytes",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name),
            )


@dataclass(frozen=True)
class B200Cta2NativeEvaluation:
    """Cooperative native result plus its compatibility reference."""

    plan: B200Cta2ExecutionPlan
    protocol_costs: Cta2ProtocolCosts
    kernel: KernelIR
    region: Any
    native: NativeModelResult
    legacy: GemmLegacyEvaluation
    participant_footprints: Tuple[Cta2ParticipantFootprint, ...]
    provenance: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", freeze_attrs(self.provenance))
        footprints = tuple(self.participant_footprints)
        if len(footprints) != 2 or {item.rank for item in footprints} != {0, 1}:
            raise ModelingValidationError(
                "CTA2 evaluation needs exactly one footprint per participant"
            )
        object.__setattr__(self, "participant_footprints", footprints)

    @property
    def body_delta_s(self) -> float:
        return self.native.kernel_body_s - float(self.legacy.legacy_result.total_latency)

    @property
    def body_ratio(self) -> float:
        baseline = float(self.legacy.legacy_result.total_latency)
        return self.native.kernel_body_s / baseline if baseline > 0.0 else math.inf


def _build_protocol_program(
    evaluation: GemmLegacyEvaluation,
    plan: B200Cta2ExecutionPlan,
    options: B200Cta2NativeOptions,
    compatibility_kernel: KernelIR,
) -> Tuple[KernelIR, Any]:
    spec = evaluation.spec
    tm, tn, tk = spec.tb_shape
    levels = spec.legacy_mem_levels()
    dtype = "bf16" if levels["in1"][-1] == 2 else "uint8"
    source_params = dict(compatibility_kernel.params)
    accumulator_dtype = source_params.get("accumulator_dtype", "fp32")
    if not isinstance(accumulator_dtype, str) or not accumulator_dtype:
        raise ModelingValidationError("CTA2 accumulator_dtype must be a string")
    super_m = tm * (2 if plan.cluster_axis == "M" else 1)
    super_n = tn * (2 if plan.cluster_axis == "N" else 1)
    cluster = (2, 1, 1) if plan.cluster_axis == "M" else (1, 2, 1)
    work_grid = evaluation.wave.work_grid + (1,)
    physical_grid = (
        work_grid[0] * cluster[0],
        work_grid[1] * cluster[1],
        1,
    )
    kernel = Kernel(
        "b200_cta2_protocol",
        params={
            "shape": spec.op_shape,
            "tile": spec.tb_shape,
            "stages": spec.stage_num,
            "cluster_axis": plan.cluster_axis,
            "plan_source": plan.source,
            "plan_kind": options.plan_kind,
        },
    )
    with kernel.launch(
        name="main",
        work_grid=work_grid,
        physical_grid=physical_grid,
        threads=128,
        cluster=cluster,
        resident_ctas=1,
        scheduler="static",
    ) as launch:
        with launch.periodic(
            "ko", iterations=int(math.ceil(spec.op_shape[2] / tk)), stages=spec.stage_num
        ) as loop:
            buffers = {}
            for route in plan.routes:
                for rank in (0, 1):
                    buffers[(route.operand, rank)] = loop.buffer(
                        "%s_rank%d" % (route.operand, rank),
                        scope="smem",
                        shape=(int(route.local_bytes[rank]),),
                        dtype="uint8",
                        slots=spec.stage_num,
                        execution_scope="cta",
                    )
            accumulator = loop.buffer(
                "C_pair",
                scope="tmem",
                shape=(super_m, super_n),
                dtype=accumulator_dtype,
                slots=1,
                execution_scope="cta_group",
            )

            writers = {}
            main_phases = []
            for rank in (0, 1):
                rank_routes = [
                    route
                    for route in plan.routes
                    if route.global_load_bytes[rank] > 0.0
                ]
                if rank_routes:
                    with loop.actor("producer%d" % rank) as producer:
                        produced = []
                        for route in rank_routes:
                            destinations = [buffers[(route.operand, rank)]]
                            if route.mode == "multicast":
                                destinations.append(
                                    buffers[(route.operand, route.target_rank)]
                                )
                            phase = producer.load(
                                "load_%s_rank%d" % (route.operand, rank),
                                bytes=route.global_load_bytes[rank],
                                source="ddr",
                                destination="smem",
                                engine="tma",
                                timing=None,
                                writes=tuple(destinations),
                                attrs={"operand": route.operand, "rank": rank, "dtype": dtype},
                            )
                            produced.append(phase)
                            main_phases.append(phase)
                            writers[(route.operand, rank)] = phase
                            if route.mode == "multicast":
                                writers[(route.operand, route.target_rank)] = phase
                        producer.sequence(
                            *(phase.at(0) for phase in produced),
                            order="issue",
                            resource="tma_rank%d" % rank,
                        )

            transfer_phases = []
            # TMA multicast is one source transaction with multiple SMEM
            # destinations and is already represented by the source load.
            # Only an explicit peer-SMEM access creates a second transfer phase.
            transferring = [
                route for route in plan.routes if route.mode == "peer_smem"
            ]
            if transferring:
                with loop.actor("cluster_link") as link:
                    for route in transferring:
                        source = buffers[(route.operand, route.source_rank)]
                        target = buffers[(route.operand, route.target_rank)]
                        phase = link.phase(
                            "transfer_%s" % route.operand,
                            work=Work(
                                "cluster_transfer",
                                bytes=route.transfer_bytes,
                                attrs={
                                    "kind": route.transfer_kind,
                                    "source_rank": route.source_rank,
                                    "target_rank": route.target_rank,
                                },
                            ),
                            timing=None,
                            reads=(source,),
                            writes=(target,),
                        )
                        loop.after(
                            writers[(route.operand, route.source_rank)].done,
                            phase.start,
                            name="%s_source_ready" % route.operand,
                        )
                        transfer_phases.append(phase)
                        main_phases.append(phase)
                        writers[(route.operand, route.target_rank)] = phase
                    link.sequence(
                        *(phase.at(0) for phase in transfer_phases),
                        order="issue",
                        resource=(
                            "cluster_link"
                            if options.protocol_costs.transfer_pricing
                            == "explicit_link"
                            else None
                        ),
                    )

            with loop.actor("pair_control") as control:
                ready = control.phase(
                    "cluster_ready",
                    work=Work(
                        "cluster_barrier",
                        attrs={"protocol": plan.load_ready_protocol},
                    ),
                    timing=None,
                )
                issue = control.phase(
                    "pair_issue",
                    work=Work(
                        "cluster_issue",
                        attrs={"leader_rank": plan.leader_rank},
                    ),
                    timing=None,
                )
                consumed = control.phase(
                    "pair_inputs_consumed",
                    work=Work(
                        "cluster_barrier",
                        attrs={"protocol": plan.input_release_protocol},
                    ),
                    timing=None,
                )
            main_phases.extend((ready, issue, consumed))
            for writer in writers.values():
                loop.after(writer.done, ready.start, name="all_operands_ready")
            loop.after(ready.done, issue.start, name="barrier_before_pair_issue")

            with loop.actor("pair_tensor") as tensor:
                mma = tensor.compute(
                    "pair_mma",
                    flops=float(2 * super_m * super_n * tk),
                    engine="tensor",
                    op="mma",
                    timing=None,
                    reads=tuple(buffers.values()) + (accumulator,),
                    writes=(accumulator,),
                    attrs={
                        "protocol": plan.pair_tensor_protocol,
                        "cta_group": 2,
                        "leader_rank": plan.leader_rank,
                    },
                )
            main_phases.append(mma)
            loop.after(issue.done, mma.start, name="pair_issue_to_mma")
            loop.after(
                issue.done,
                consumed.start,
                name="pair_issue_to_input_consumed",
            )
            state = loop.state("accumulator", storage=accumulator)
            loop.carry(state, source=mma.done, target=mma.start, distance=1)
            for key, buffer in buffers.items():
                writer = writers[key]
                loop.pipeline_buffer(
                    buffer,
                    acquire=writer.start,
                    release=consumed.done,
                    capacity=spec.stage_num,
                )

        with launch.periodic("epilogue", iterations=1, stages=1) as epilogue:
            with epilogue.actor("pair_store") as store_actor:
                store = store_actor.store(
                    "store_pair",
                    bytes=float(super_m * super_n * levels["out1"][-1]),
                    source="tmem",
                    destination="ddr",
                    engine="tma",
                    timing=None,
                    attrs={"partitioned_across_ranks": True},
                )

    ir = kernel.build()
    phases = {phase.name: phase for phase in ir.periodic_loop("ko").phases}
    main_order = tuple(
        PhaseRegion(phases[phase.name]) for phase in main_phases
    )
    main_region = LoopRegion(
        "ko_region",
        trip_count=int(math.ceil(spec.op_shape[2] / tk)),
        body=SequenceRegion("ko_body", main_order),
        periodic_axis=PeriodicAxisIR(
            loop_name="ko",
            ii_mode=options.ii_mode,
            boundary_anchor_phase="pair_mma",
            search_config=options.search_config,
        ),
    )
    store_phase = ir.periodic_loop("epilogue").phases[0]
    root = SequenceRegion(
        "cta2_kernel",
        (main_region, PhaseRegion(store_phase)),
    )
    return ir, root


class B200Cta2LegacyBoundOracle:
    """Bind a two-participant DAG to existing calibrated resource costs."""

    supports_cooperative_cluster = True
    cooperative_cluster_size = 2
    consumes_cluster_operand_plan = True
    consumes_pair_tensor_throughput = True
    consumes_per_sm_local_smem = True
    consumes_cluster_issue_latency = True

    def __init__(
        self,
        evaluation: GemmLegacyEvaluation,
        plan: B200Cta2ExecutionPlan,
        protocol_costs: Cta2ProtocolCosts,
        context: Optional[Any] = None,
    ) -> None:
        self.evaluation = evaluation
        self.plan = plan
        self.protocol_costs = protocol_costs
        self.context = context
        self.consumes_peer_smem_traffic = plan.has_peer_smem
        detail = evaluation.legacy_result.pipeline_detail
        if detail is None:
            raise ModelingValidationError("CTA2 legacy binding needs pipeline detail")
        self.memory_s = float(detail.mem_time_per_iter)
        self.tensor_s = float(detail.compute_time_per_iter)
        depth = max(evaluation.spec.stage_num - 1, 0)
        self.store_s = max(
            float(detail.epilogue_time) - depth * self.tensor_s,
            0.0,
        )
        rank_bytes = tuple(
            sum(route.global_load_bytes[rank] for route in plan.routes)
            for rank in (0, 1)
        )
        self.rank_bytes = rank_bytes
        self.busiest_rank_bytes = max(rank_bytes)
        if self.memory_s <= 0.0 or self.tensor_s <= 0.0 or self.busiest_rank_bytes <= 0.0:
            raise ModelingValidationError("CTA2 calibrated resource costs must be positive")

    def bind_context(self, context: Any) -> "B200Cta2LegacyBoundOracle":
        return B200Cta2LegacyBoundOracle(
            self.evaluation,
            self.plan,
            self.protocol_costs,
            context=context,
        )

    @staticmethod
    def _timing(latency: float, resource: Optional[str] = None) -> Timing:
        if latency <= 0.0:
            return Timing(0.0)
        resources = () if resource is None else (ResourceTiming(resource, latency),)
        return Timing(latency, resources)

    @staticmethod
    def _timing_with_service(
        latency: float,
        resource: str,
        service_time: float,
    ) -> Timing:
        if latency <= 0.0 or service_time <= 0.0:
            return Timing(0.0)
        return Timing(
            latency,
            (ResourceTiming(resource, service_time),),
        )

    def resolve(self, phase: Phase) -> Timing:
        name = phase.name
        if name.startswith("load_"):
            attrs = dict(phase.work.attrs)
            rank = int(attrs["rank"])
            duration = self.memory_s * phase.work.bytes / self.busiest_rank_bytes
            return self._timing_with_service(
                duration + self.protocol_costs.tma_fixed_latency_s,
                "tma_rank%d" % rank,
                duration,
            )
        if name.startswith("transfer_"):
            operand = name.split("_", 1)[1]
            route = self.plan.route_map[operand]
            duration = self.protocol_costs.transfer_time(route)
            return self._timing(duration, "cluster_link" if duration > 0.0 else None)
        if name == "cluster_ready":
            duration = (
                self.context.topology.cluster_barrier_s
                if self.context is not None
                else self.protocol_costs.cluster_barrier_s
            )
            return self._timing(duration, "cluster_control" if duration > 0.0 else None)
        if name == "pair_issue":
            duration = (
                self.context.topology.cluster_issue_s
                if self.context is not None
                else self.protocol_costs.cluster_issue_s
            )
            return self._timing(duration, "pair_issue" if duration > 0.0 else None)
        if name == "pair_inputs_consumed":
            duration = self.protocol_costs.input_consumed_latency_s
            if duration is None:
                duration = self.tensor_s
            if duration > self.tensor_s + 1.0e-15:
                raise ModelingValidationError(
                    "input-consumed latency cannot exceed pair MMA completion"
                )
            return self._timing(duration)
        if name == "pair_mma":
            return self._timing(self.tensor_s, "tensor_pair")
        if name == "store_pair":
            return self._timing(self.store_s, "cluster_store")
        raise ModelingValidationError("unknown CTA2 protocol phase %r" % name)


def model_b200_cta2_native(
    compatibility_kernel: KernelIR,
    arch: Any,
    options: Optional[B200Cta2NativeOptions] = None,
) -> B200Cta2NativeEvaluation:
    """Build and execute an explicit two-participant cooperative DAG.

    The input is the strictly validated compatibility KernelIR so legacy and
    protocol predictions share exactly the same shape/tile/stage/grid facts.
    """

    if options is None:
        options = B200Cta2NativeOptions()
    if not isinstance(options, B200Cta2NativeOptions):
        raise ModelingValidationError("CTA2 native options have the wrong type")
    if getattr(arch, "core", None) != "B200":
        raise ModelingValidationError("CTA2 native prototype currently requires B200")
    legacy = evaluate_gemm_legacy(compatibility_kernel, arch)
    if legacy.wave.effective_mma_type != "utcmma_cta2":
        raise ModelingValidationError("kernel does not select an effective B200 CTA2 tile")
    plan = B200Cta2ExecutionPlan.from_legacy_spec(
        legacy,
        plan_kind=options.plan_kind,
        source=options.plan_source,
    )
    if (
        plan.has_peer_smem
        and options.protocol_costs.transfer_pricing == "explicit_link"
        and options.protocol_costs.transfer_bandwidth_bytes_per_s is None
        and options.protocol_costs.transfer_fixed_latency_s <= 0.0
    ):
        raise ModelingValidationError("CTA2 payload transfer is unpriced")
    if (
        options.protocol_costs.transfer_pricing == "explicit_link"
        and not plan.has_peer_smem
    ):
        raise ModelingValidationError(
            "explicit_link pricing is only valid for an explicit peer_smem route; "
            "TMA multicast stays part of the TMA source transaction"
        )
    kernel, region = _build_protocol_program(
        legacy, plan, options, compatibility_kernel
    )
    topology = LaunchTopology.cooperative_cluster(
        2,
        operand_plan=plan.topology_operand_plan,
        tma_multicast=plan.has_tma_multicast,
        peer_smem_resources=("cluster_link",) if plan.has_peer_smem else tuple(),
        cluster_barrier_s=options.protocol_costs.cluster_barrier_s,
        cluster_issue_s=options.protocol_costs.cluster_issue_s,
    )
    oracle = B200Cta2LegacyBoundOracle(legacy, plan, options.protocol_costs)
    native = model_native(
        kernel,
        region,
        arch,
        NativeModelOptions(
            ii_mode=options.ii_mode,
            boundary_policy=options.boundary_policy,
            wave_policy=options.wave_policy,
            launch_topology=topology,
            resident_ctas_per_sm=1,
            multi_cta_policy="reject",
            liveness_policy="require_witness",
            kernel_launch_s=options.kernel_launch_s,
            host_dispatch_s=options.host_dispatch_s,
        ),
        oracle=oracle,
    )
    params = dict(compatibility_kernel.params)
    accumulator_bytes = DTypeSpec(params.get("accumulator_dtype", "fp32")).storage_bytes
    tm, tn, _tk = legacy.spec.tb_shape
    pair_tmem_bytes = float(2 * tm * tn * accumulator_bytes)
    participant_footprints = tuple(
        Cta2ParticipantFootprint(
            rank=rank,
            smem_bytes_per_stage=sum(
                route.local_bytes[rank] for route in plan.routes
            ),
            smem_all_stages_bytes=legacy.spec.stage_num
            * sum(route.local_bytes[rank] for route in plan.routes),
            tmem_bytes=pair_tmem_bytes / 2.0,
        )
        for rank in (0, 1)
    )
    max_smem = max(item.smem_all_stages_bytes for item in participant_footprints)
    max_tmem = max(item.tmem_bytes for item in participant_footprints)
    native = replace(
        native,
        liveness=LivenessReport(
            status=FeasibilityStatus.UNKNOWN,
            peaks=(
                StoragePeak(
                    "smem",
                    "physical_cta",
                    max_smem,
                    max_smem,
                    max_smem,
                    "explicit_participant_static_allocation",
                ),
                StoragePeak(
                    "tmem",
                    "physical_sm",
                    max_tmem,
                    max_tmem,
                    max_tmem,
                    "explicit_participant_partition",
                ),
            ),
            diagnostics=(
                "CTA2 storage is reported per physical participant; expanded-DAG "
                "group buffers are not compared with one-SM capacity",
                "register allocation, TMEM allocator protocol, and cooperative "
                "admission remain unproven",
                "report-only: this result is not a liveness feasibility guard",
            ),
            witness_ii_s=native.liveness.witness_ii_s,
            guard_eligible=False,
        ),
    )
    return B200Cta2NativeEvaluation(
        plan=plan,
        protocol_costs=options.protocol_costs,
        kernel=kernel,
        region=region,
        native=native,
        legacy=legacy,
        participant_footprints=participant_footprints,
        provenance=(
            ("cost_binding", "legacy_per_iteration_bounds_explicit_protocol_dag"),
            ("independent_hardware_calibration", False),
            ("tail_memory_repricing", "fixed_calibrated_per_logical_slot"),
            ("generic_dsm_bandwidth", "not_created"),
            ("pair_tensor_protocol", plan.pair_tensor_protocol),
            ("load_ready_protocol", plan.load_ready_protocol),
            ("input_release_protocol", plan.input_release_protocol),
            (
                "pair_tensor_internal_operand_coordination",
                "priced_by_tensor_pair_resource_not_generic_dsm_bytes",
            ),
            ("completion_protocol", plan.completion_protocol),
            ("plan_source", plan.source),
            ("payload_transfer_bytes_per_iteration", plan.payload_transfer_bytes_per_iteration),
            ("transfer_pricing", options.protocol_costs.transfer_pricing),
            ("tma_fixed_latency_s", options.protocol_costs.tma_fixed_latency_s),
            ("boundary_policy", options.boundary_policy),
        ),
    )


__all__ = [
    "B200Cta2ExecutionPlan",
    "B200Cta2LegacyBoundOracle",
    "B200Cta2NativeEvaluation",
    "B200Cta2NativeOptions",
    "Cta2OperandRoute",
    "Cta2ParticipantFootprint",
    "Cta2PhysicalTransfer",
    "Cta2ProtocolCosts",
    "model_b200_cta2_native",
]
