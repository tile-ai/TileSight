"""Versioned JSON snapshots of ``Program`` (round-trip) and results (export)."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

from ..ir import ResourceTiming, Timing
from ..ops.schema import (
    DTypeSpec, ElementwiseOpSpec, EngineSpec, GemmOpSpec, ReduceOpSpec, ReductionAxisSpec,
    ScalarWorkSpec, TensorOperandSpec, TensorResultSpec,
)
from .contract import (
    Actor, Buffer, BufferSlots, Carry, Compute, ContractError, Dependency, EventRef, StorageGroup,
    FromWorkGroup, Launch, LaunchEdge, Load, Loop, PROGRAM_SCHEMA_VERSION, Persistent, mapper_from_canonical,
    Pipeline, Program, Projection, ResourceOrder, Sequence, Store, Sync, TileAccess,
    WorkAxis, WorkGroup, WorkRun, WorkUnits, is_op,
)


# -- encoding ---------------------------------------------------------------


def _timing(timing: Optional[Timing]) -> Optional[Dict[str, Any]]:
    if timing is None:
        return None
    return {
        "latency_s": timing.latency,
        "resources": [
            {"resource": r.resource, "service_time_s": r.service_time, "offset_s": r.offset}
            for r in timing.resources
        ],
    }


def _access(access: Optional[TileAccess]) -> Optional[Dict[str, Any]]:
    return None if access is None else access.canonical()


def _scalar_work(item: ScalarWorkSpec) -> Dict[str, Any]:
    return {
        "op_class": item.op_class, "count": item.count, "engine": item.engine.requested,
        "dtype": item.dtype.name, "semantic_flops_per_op": item.semantic_flops_per_op,
        "issue_equivalent_flops_per_op": item.issue_equivalent_flops_per_op,
        "attrs": dict(item.attrs),
    }


def _operand(item: TensorOperandSpec) -> Dict[str, Any]:
    return {
        "name": item.name, "shape": list(item.shape), "storage_dtype": item.storage_dtype.name,
        "access": item.access, "index_map": dict(item.index_map),
    }


def _spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, GemmOpSpec):
        return {
            "type": "GemmOpSpec", "op_id": spec.op_id, "m": spec.m, "n": spec.n, "k": spec.k,
            "batch": spec.batch, "a_storage_dtype": spec.a_storage_dtype.name,
            "b_storage_dtype": spec.b_storage_dtype.name,
            "result_storage_dtype": spec.result_storage_dtype.name,
            "compute_dtype": spec.compute_dtype.name,
            "accumulation_dtype": spec.accumulation_dtype.name,
            "engine": spec.engine.requested, "attrs": dict(spec.attrs),
        }
    if isinstance(spec, ElementwiseOpSpec):
        return {
            "type": "ElementwiseOpSpec", "op_id": spec.op_id,
            "operands": [_operand(x) for x in spec.operands],
            "result": {"name": spec.result.name, "shape": list(spec.result.shape),
                       "storage_dtype": spec.result.storage_dtype.name},
            "operations": [_scalar_work(x) for x in spec.operations],
            "compute_dtype": spec.compute_dtype.name,
            "accumulation_dtype": spec.accumulation_dtype.name, "attrs": dict(spec.attrs),
        }
    if isinstance(spec, ReduceOpSpec):
        return {
            "type": "ReduceOpSpec", "op_id": spec.op_id,
            "operands": [_operand(x) for x in spec.operands],
            "result": {"name": spec.result.name, "shape": list(spec.result.shape),
                       "storage_dtype": spec.result.storage_dtype.name},
            "reduction_axes": [
                {"axis": a.axis, "extent": a.extent, "tile_step": a.tile_step,
                 "operand_names": list(a.operand_names)} for a in spec.reduction_axes
            ],
            "reducer": spec.reducer, "work": [_scalar_work(x) for x in spec.work],
            "compute_dtype": spec.compute_dtype.name,
            "accumulation_dtype": spec.accumulation_dtype.name,
            "keepdims": spec.keepdims, "attrs": dict(spec.attrs),
        }
    raise ContractError("snapshot: unsupported semantic spec %r" % type(spec).__name__)


def _op(op: Any) -> Dict[str, Any]:
    base = {
        "op_id": op.op_id, "actor": op.actor, "reads": list(op.reads), "writes": list(op.writes),
        "timing": _timing(op.timing), "latency_extra_s": op.latency_extra_s,
        "metadata": dict(op.metadata),
    }
    if isinstance(op, (Load, Store)):
        base.update({
            "type": "Load" if isinstance(op, Load) else "Store",
            "source": op.source, "destination": op.destination, "engine": op.engine,
            "access": _access(op.access), "bytes": op.bytes,
        })
    elif isinstance(op, Compute):
        base.update({
            "type": "Compute", "spec": _spec(op.spec), "engine": op.engine,
            "effective_fraction": op.effective_fraction,
        })
    elif isinstance(op, Sync):
        base.update({"type": "Sync", "sync_kind": op.sync_kind, "participants": list(op.participants)})
    else:
        raise ContractError("snapshot: unknown op %r" % (op,))
    return base


def _event(ref: EventRef) -> Dict[str, str]:
    return {"op_id": ref.op_id, "kind": ref.kind}


def _pipeline(pipeline: Pipeline) -> Dict[str, Any]:
    return {
        "type": "Pipeline",
        "actors": [
            {
                "name": a.name, "ops": [_op(op) for op in a.ops],
                "sequence": [list(item) for item in a.sequence], "order": a.order,
                "resource": a.resource, "execution_scope": a.execution_scope,
            }
            for a in pipeline.actors
        ],
        "buffers": [
            {"name": b.name, "storage": b.storage, "shape": list(b.shape), "dtype": b.dtype,
             "slots": b.slots, "execution_scope": b.execution_scope}
            for b in pipeline.buffers
        ],
        "dependencies": [
            {"source": _event(d.source), "target": _event(d.target), "distance": d.distance,
             "lag_s": d.lag_s, "name": d.name}
            for d in pipeline.dependencies
        ],
        "carries": [
            {"state": c.state, "source": _event(c.source), "target": _event(c.target),
             "distance": c.distance, "storage": c.storage, "lag_s": c.lag_s}
            for c in pipeline.carries
        ],
        "buffer_slots": [
            {"buffer": s.buffer, "acquire": _event(s.acquire), "release": _event(s.release),
             "capacity": s.capacity}
            for s in pipeline.buffer_slots
        ],
        "resource_orders": [
            {"resource": r.resource, "sequence": [list(item) for item in r.sequence]}
            for r in pipeline.resource_orders
        ],
        "storage_groups": [
            {"name": g.name, "kind": g.kind, "members": list(g.members), "slots": g.slots}
            for g in pipeline.storage_groups
        ],
        "stages": pipeline.stages,
    }


def _region(region: Any) -> Dict[str, Any]:
    if is_op(region):
        return _op(region)
    if isinstance(region, Sequence):
        return {"type": "Sequence", "name": region.name,
                "children": [_region(c) for c in region.children]}
    if isinstance(region, Loop):
        return {
            "type": "Loop", "loop_id": region.loop_id,
            "trip_count": (
                {"kind": "from_work_group"} if isinstance(region.trip_count, FromWorkGroup)
                else region.trip_count
            ),
            "body": _pipeline(region.body) if isinstance(region.body, Pipeline) else _region(region.body),
            "prologue": None if region.prologue is None else _region(region.prologue),
            "epilogue": None if region.epilogue is None else _region(region.epilogue),
            "ii_mode": region.ii_mode, "boundary_anchor_op": region.boundary_anchor_op,
        }
    raise ContractError("snapshot: unknown region %r" % (region,))


def _launch(launch: Launch) -> Dict[str, Any]:
    scheduler = (
        launch.scheduler if isinstance(launch.scheduler, str)
        else {"kind": "persistent", "partitions": [[list(r) for r in part] for part in launch.scheduler.partitions]}
    )
    return {
        "name": launch.name, "body": _region(launch.body),
        "work_axes": [{"name": a.name, "extent": a.extent} for a in launch.work_axes],
        "threads": launch.threads, "grid": None if launch.grid is None else list(launch.grid),
        "cluster": list(launch.cluster), "residency": launch.residency,
        "work_units": None if launch.work_units is None else {
            "groups": [
                {"group_id": gid, "loop_trip_counts": [list(x) for x in g.loop_trip_counts],
                 "region_variant": g.region_variant,
                 "effective_fractions": [list(x) for x in g.effective_fractions]}
                for gid, g in launch.work_units.groups
            ],
            "order": [{"group_id": r.group_id, "count": r.count} for r in launch.work_units.order],
        },
        "scheduler": scheduler,
        "dispatch_order": (
            launch.dispatch_order if isinstance(launch.dispatch_order, str) else launch.dispatch_order.canonical()
        ),
        "requires_arch": list(launch.requires_arch),
        "metadata": dict(launch.metadata),
        "buffers": [
            {"name": b.name, "storage": b.storage, "shape": list(b.shape), "dtype": b.dtype,
             "slots": b.slots, "execution_scope": b.execution_scope}
            for b in launch.buffers
        ],
        "storage_groups": [
            {"name": g.name, "kind": g.kind, "members": list(g.members), "slots": g.slots}
            for g in launch.storage_groups
        ],
    }


def _strip_cost_fields(value: Any) -> Any:
    """Drop timing overrides and metadata so only the workload identity remains."""

    if isinstance(value, dict):
        return {
            key: _strip_cost_fields(item)
            for key, item in value.items()
            if key not in ("timing", "latency_extra_s", "metadata")
        }
    if isinstance(value, list):
        return [_strip_cost_fields(item) for item in value]
    return value


def launch_workload_digest(launch: Launch) -> str:
    """SHA-256 of the launch's shapes, dtypes, work structure, and grouping."""

    import hashlib
    import json

    payload = json.dumps(_strip_cost_fields(_launch(launch)), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def program_workload_digest(program: Program) -> str:
    import hashlib
    import json

    payload = json.dumps(
        {"name": program.name, "launches": [_strip_cost_fields(_launch(l)) for l in program.launches],
         "edges": [[e.source, e.target] for e in program.edges]},
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def program_to_snapshot(program: Program) -> Dict[str, Any]:
    return {
        "schema": PROGRAM_SCHEMA_VERSION,
        "name": program.name,
        "launches": [_launch(l) for l in program.launches],
        "edges": [{"source": e.source, "target": e.target, "gap_s": e.gap_s} for e in program.edges],
        "metadata": dict(program.metadata),
    }


# -- decoding ---------------------------------------------------------------


def _timing_from(data: Optional[Mapping[str, Any]]) -> Optional[Timing]:
    if data is None:
        return None
    return Timing(
        data["latency_s"],
        tuple(ResourceTiming(r["resource"], r["service_time_s"], r.get("offset_s", 0.0))
              for r in data.get("resources", [])),
    )


def _access_from(data: Optional[Mapping[str, Any]]) -> Optional[TileAccess]:
    if data is None:
        return None
    return TileAccess(
        tensor=data["tensor"], index_map=Projection(tuple(data["index_map"]["work_axes"])),
        tile_shape=tuple(data["tile_shape"]), element_bytes=data["element_bytes"],
        valid_shape=None if data.get("valid_shape") is None else tuple(data["valid_shape"]),
        tensor_shape=None if data.get("tensor_shape") is None else tuple(data["tensor_shape"]),
        transaction_bytes=data.get("transaction_bytes"), allocation_bytes=data.get("allocation_bytes"),
        access_id=data.get("access_id"),
    )


def _scalar_work_from(data: Mapping[str, Any]) -> ScalarWorkSpec:
    return ScalarWorkSpec(
        data["op_class"], data["count"], EngineSpec(data["engine"]), DTypeSpec(data["dtype"]),
        data.get("semantic_flops_per_op", 0.0), data.get("issue_equivalent_flops_per_op", 0.0),
        data.get("attrs", {}),
    )


def _operand_from(data: Mapping[str, Any]) -> TensorOperandSpec:
    return TensorOperandSpec(
        data["name"], tuple(data["shape"]), DTypeSpec(data["storage_dtype"]),
        data.get("access", "identity"), data.get("index_map", {}),
    )


def _spec_from(data: Mapping[str, Any]) -> Any:
    kind = data["type"]
    if kind == "GemmOpSpec":
        return GemmOpSpec(
            op_id=data["op_id"], m=data["m"], n=data["n"], k=data["k"], batch=data["batch"],
            a_storage_dtype=DTypeSpec(data["a_storage_dtype"]),
            b_storage_dtype=DTypeSpec(data["b_storage_dtype"]),
            result_storage_dtype=DTypeSpec(data["result_storage_dtype"]),
            compute_dtype=DTypeSpec(data["compute_dtype"]),
            accumulation_dtype=DTypeSpec(data["accumulation_dtype"]),
            engine=EngineSpec(data["engine"]), attrs=data.get("attrs", {}),
        )
    result = data["result"]
    if kind == "ElementwiseOpSpec":
        return ElementwiseOpSpec(
            op_id=data["op_id"], operands=tuple(_operand_from(x) for x in data["operands"]),
            result=TensorResultSpec(result["name"], tuple(result["shape"]), DTypeSpec(result["storage_dtype"])),
            operations=tuple(_scalar_work_from(x) for x in data["operations"]),
            compute_dtype=DTypeSpec(data["compute_dtype"]),
            accumulation_dtype=DTypeSpec(data["accumulation_dtype"]), attrs=data.get("attrs", {}),
        )
    if kind == "ReduceOpSpec":
        return ReduceOpSpec(
            op_id=data["op_id"], operands=tuple(_operand_from(x) for x in data["operands"]),
            result=TensorResultSpec(result["name"], tuple(result["shape"]), DTypeSpec(result["storage_dtype"])),
            reduction_axes=tuple(
                ReductionAxisSpec(a["axis"], a["extent"], a["tile_step"], tuple(a["operand_names"]))
                for a in data["reduction_axes"]
            ),
            reducer=data.get("reducer", "sum"), work=tuple(_scalar_work_from(x) for x in data["work"]),
            compute_dtype=DTypeSpec(data["compute_dtype"]),
            accumulation_dtype=DTypeSpec(data["accumulation_dtype"]),
            keepdims=data.get("keepdims", False), attrs=data.get("attrs", {}),
        )
    raise ContractError("snapshot: unknown spec type %r" % kind)


def _op_from(data: Mapping[str, Any]) -> Any:
    kind = data["type"]
    common = dict(
        op_id=data["op_id"], actor=data.get("actor", "default"),
        reads=tuple(data.get("reads", [])), writes=tuple(data.get("writes", [])),
        timing=_timing_from(data.get("timing")), latency_extra_s=data.get("latency_extra_s"),
        metadata=data.get("metadata", {}),
    )
    if kind in ("Load", "Store"):
        cls = Load if kind == "Load" else Store
        return cls(
            source=data["source"], destination=data["destination"], engine=data.get("engine", "unbound"),
            access=_access_from(data.get("access")), bytes=data.get("bytes"), **common
        )
    if kind == "Compute":
        return Compute(
            spec=_spec_from(data["spec"]), engine=data.get("engine"),
            effective_fraction=data.get("effective_fraction", 1.0), **common
        )
    if kind == "Sync":
        return Sync(sync_kind=data.get("sync_kind", "barrier"),
                    participants=tuple(data.get("participants", [])), **common)
    raise ContractError("snapshot: unknown op type %r" % kind)


def _event_from(data: Mapping[str, Any]) -> EventRef:
    return EventRef(data["op_id"], data.get("kind", "done"))


def _pipeline_from(data: Mapping[str, Any]) -> Pipeline:
    return Pipeline(
        actors=tuple(
            Actor(
                name=a["name"], ops=tuple(_op_from(op) for op in a["ops"]),
                sequence=tuple((s[0], s[1]) for s in a.get("sequence", [])),
                order=a.get("order", "issue"), resource=a.get("resource"),
                execution_scope=a.get("execution_scope", "unspecified"),
            )
            for a in data["actors"]
        ),
        buffers=tuple(
            Buffer(b["name"], b["storage"], tuple(b.get("shape", [])), b.get("dtype", ""),
                   b.get("slots", 1), b.get("execution_scope", "cta"))
            for b in data.get("buffers", [])
        ),
        dependencies=tuple(
            Dependency(_event_from(d["source"]), _event_from(d["target"]), d.get("distance", 0),
                       d.get("lag_s", 0.0), d.get("name", ""))
            for d in data.get("dependencies", [])
        ),
        carries=tuple(
            Carry(c["state"], _event_from(c["source"]), _event_from(c["target"]),
                  c.get("distance", 1), c.get("storage"), c.get("lag_s", 0.0))
            for c in data.get("carries", [])
        ),
        buffer_slots=tuple(
            BufferSlots(s["buffer"], _event_from(s["acquire"]), _event_from(s["release"]), s.get("capacity"))
            for s in data.get("buffer_slots", [])
        ),
        resource_orders=tuple(
            ResourceOrder(r["resource"], tuple((s[0], s[1]) for s in r["sequence"]))
            for r in data.get("resource_orders", [])
        ),
        storage_groups=tuple(
            StorageGroup(g["name"], g["kind"], tuple(g["members"]), g.get("slots", 1))
            for g in data.get("storage_groups", [])
        ),
        stages=data.get("stages"),
    )


def _region_from(data: Mapping[str, Any]) -> Any:
    kind = data["type"]
    if kind == "Sequence":
        return Sequence(tuple(_region_from(c) for c in data["children"]), data.get("name"))
    if kind == "Loop":
        trip = data["trip_count"]
        body = data["body"]
        return Loop(
            loop_id=data["loop_id"],
            trip_count=FromWorkGroup() if isinstance(trip, Mapping) else trip,
            body=_pipeline_from(body) if body.get("type") == "Pipeline" else _region_from(body),
            prologue=None if data.get("prologue") is None else _region_from(data["prologue"]),
            epilogue=None if data.get("epilogue") is None else _region_from(data["epilogue"]),
            ii_mode=data.get("ii_mode", "inherit"), boundary_anchor_op=data.get("boundary_anchor_op"),
        )
    return _op_from(data)


def _launch_from(data: Mapping[str, Any]) -> Launch:
    scheduler = data.get("scheduler", "static_dispatch")
    if isinstance(scheduler, Mapping):
        scheduler = Persistent(tuple(tuple(tuple(r) for r in part) for part in scheduler["partitions"]))
    units = data.get("work_units")
    work_units = None
    if units is not None:
        work_units = WorkUnits(
            groups=tuple(
                (g["group_id"], WorkGroup(
                    tuple((x[0], x[1]) for x in g.get("loop_trip_counts", [])),
                    g.get("region_variant"),
                    tuple((x[0], x[1]) for x in g.get("effective_fractions", [])),
                ))
                for g in units["groups"]
            ),
            order=tuple(WorkRun(r["group_id"], r["count"]) for r in units["order"]),
        )
    dispatch = mapper_from_canonical(data.get("dispatch_order", "linear_block_id"))
    return Launch(
        name=data["name"], body=_region_from(data["body"]),
        work_axes=tuple(WorkAxis(a["name"], a["extent"]) for a in data["work_axes"]),
        threads=data.get("threads", 128), grid=None if data.get("grid") is None else tuple(data["grid"]),
        cluster=tuple(data.get("cluster", [1, 1, 1])), residency=data.get("residency", "auto"),
        work_units=work_units, scheduler=scheduler,
        dispatch_order=dispatch,
        requires_arch=tuple(data.get("requires_arch", [])), metadata=data.get("metadata", {}),
        buffers=tuple(
            Buffer(b["name"], b["storage"], tuple(b.get("shape", [])), b.get("dtype", ""),
                   b.get("slots", 1), b.get("execution_scope", "cta"))
            for b in data.get("buffers", [])
        ),
        storage_groups=tuple(
            StorageGroup(g["name"], g["kind"], tuple(g["members"]), g.get("slots", 1))
            for g in data.get("storage_groups", [])
        ),
    )


def program_from_snapshot(data: Mapping[str, Any]) -> Program:
    schema = data.get("schema")
    if schema != PROGRAM_SCHEMA_VERSION:
        raise ContractError("snapshot schema %r is not %r" % (schema, PROGRAM_SCHEMA_VERSION))
    return Program(
        launches=tuple(_launch_from(l) for l in data["launches"]),
        edges=tuple(LaunchEdge(e["source"], e["target"], e.get("gap_s", 0.0)) for e in data.get("edges", [])),
        name=data.get("name", "program"), metadata=data.get("metadata", {}),
    )


__all__ = ["launch_workload_digest", "program_from_snapshot", "program_to_snapshot", "program_workload_digest"]
