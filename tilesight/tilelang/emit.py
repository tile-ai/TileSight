"""Neutral tile program -> TileSight ``KernelBuilder`` calls."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .errors import Refusal
from .translate import Buf, CtaTrips, Op, Program, Slice


def _count(shape: Tuple[int, ...]) -> int:
    total = 1
    for extent in shape:
        total *= extent
    return total


def _engine(op: Op, storage: str, mechanisms: Tuple[str, ...], notes: List[str]) -> str:
    """Transfer engine: explicit in the IR, else the single async family the kernel source shows."""

    if op.engine:
        return op.engine
    if storage == "register":
        return "ldst"
    families = [m for m in ("tma", "cp_async") if m in mechanisms]
    if len(families) == 1:
        return families[0]
    if families:
        notes.append("%s: kernel source shows %s; engine not attributable per copy, modeled as ldst"
                     % (op.name, " and ".join(families)))
    return "ldst"


def build(program: Program, sight: Any, *, residency: Any = "auto", mechanisms: Tuple[str, ...] = (),
          metadata: Optional[Dict[str, Any]] = None) -> Any:
    kb = sight.KernelBuilder("main", grid=dict(program.grid), threads=program.threads, residency=residency,
                             metadata=dict(metadata or {}))
    tiles = sight.tiles
    tensors = {name: kb.tensor(name, t.shape, t.dtype) for name, t in program.tensors.items()}
    refs: Dict[str, Any] = {}

    def ref(name: str) -> Any:
        if name not in refs:
            buf: Buf = program.buffers[name]
            if buf.storage == "smem":
                refs[name] = kb.shared(name, buf.shape, buf.dtype, stages=buf.slots)
            else:
                refs[name] = kb.fragment(name, buf.shape, buf.dtype, carried=buf.carried)
        return refs[name]

    def view(sl: Slice) -> Any:
        items = []
        for item in sl.items:
            if item[0] == "tile":
                items.append(tiles(item[1]))
            elif item[0] == "pick":
                items.append(tuple(item[1]))
            else:
                items.append(slice(None))
        if all(isinstance(item, tuple) for item in items):
            # one element per CTA: walking the last single-axis pick in tiles of 1 reads the same element
            last = next((i for i in range(len(items) - 1, -1, -1) if len(items[i]) == 1), None)
            if last is None:
                raise Refusal("INDEX_UNSUPPORTED", "one element of %s picked only by composite axes" % sl.tensor)
            items[last] = tiles(items[last][0])
        return tensors[sl.tensor][tuple(items)]

    def kwargs(op: Op) -> Dict[str, Any]:
        out: Dict[str, Any] = {"window_offset": op.window} if op.window else {}
        if op.actor:
            out["actor"] = op.actor
        return out

    def operand(name: str, op: Op) -> Any:
        if name in op.row_inputs:
            return kb.row_state(ref(name))
        if name in op.mapped_inputs:
            return kb.mapped(ref(name), {"kind": "affine", "elements": int(_count(program.buffers[name].shape))})
        return ref(name)

    def emit(op: Op) -> None:
        if op.kind == "load":
            dst = program.buffers[op.dst]
            tensor = program.tensors[op.src.tensor]
            engine = _engine(op, dst.storage, mechanisms, program.notes)
            if tensor.dtype != dst.dtype:
                if dst.storage != "register":
                    raise Refusal("DTYPE_CONVERSION_UNSUPPORTED", "%s converts %s to %s into shared memory"
                                  % (op.name, tensor.dtype, dst.dtype))
                raw = kb.fragment(op.dst + "__as_" + tensor.dtype, dst.shape, tensor.dtype)
                kb.copy(view(op.src), raw, engine=engine, name=op.name, **kwargs(op))
                kb.cast(raw, ref(op.dst), name=op.name + "_convert", **kwargs(op))
            else:
                kb.copy(view(op.src), ref(op.dst), engine=engine, name=op.name, **kwargs(op))
        elif op.kind == "store":
            src = program.buffers[op.src]
            tensor = program.tensors[op.dst.tensor]
            if tensor.dtype != src.dtype:
                if src.storage != "register":
                    raise Refusal("DTYPE_CONVERSION_UNSUPPORTED", "%s converts %s to %s out of shared memory"
                                  % (op.name, src.dtype, tensor.dtype))
                converted = kb.fragment(op.src + "__as_" + tensor.dtype, src.shape, tensor.dtype)
                kb.cast(ref(op.src), converted, name=op.name + "_convert", **kwargs(op))
                kb.copy(converted, view(op.dst), engine=_engine(op, "register", mechanisms, program.notes),
                        name=op.name, **kwargs(op))
            else:
                kb.copy(ref(op.src), view(op.dst), engine=_engine(op, src.storage, mechanisms, program.notes),
                        name=op.name, **kwargs(op))
        elif op.kind == "move":
            src, dst = program.buffers[op.src], program.buffers[op.dst]
            if src.storage == "register" and dst.storage == "register":
                cls = "convert" if src.dtype != dst.dtype else "mov"
                kb.elementwise(op.name, [ref(op.src)], ref(op.dst), ((cls, 1.0, "cuda"),),
                               compute_dtype="fp32", **kwargs(op))
            else:
                kb.copy(ref(op.src), ref(op.dst), engine=op.engine or "ldst", name=op.name, **kwargs(op))
        elif op.kind == "gemm":
            a, b = op.inputs
            kb.gemm(ref(a), ref(b), ref(op.output), transpose_b=op.transpose_b, name=op.name, **kwargs(op))
        elif op.kind == "elementwise":
            output = program.buffers[op.output]
            inputs = [operand(name, op) for name in op.inputs if name != op.output or not output.carried]
            kb.elementwise(op.name, inputs, ref(op.output), op.ops, compute_dtype=op.compute_dtype,
                           **kwargs(op))
        elif op.kind == "reduce":
            (source,) = op.inputs
            kb.reduce(op.name, ref(source), ref(op.output), axis=op.axis, reducer=op.reducer, ops=op.ops,
                      **kwargs(op))
        elif op.kind == "barrier":
            kb.sync(op.name, "barrier", actor=op.actor)
        else:
            raise Refusal("OPERATION_UNSUPPORTED", "cannot emit %s" % op.kind)

    def walk(items: List[Any]) -> None:
        for item in items:
            if isinstance(item, Op):
                emit(item)
                continue
            trips = kb.from_work_group() if isinstance(item.trips, CtaTrips) else item.trips
            with kb.loop(item.name, trips, stages=item.stages) as loop:
                if item.prologue:
                    with loop.prologue():
                        walk(item.prologue)
                walk(item.body)
                windows = {op.name: op.window for op in item.body if isinstance(op, Op)}
                for role, names in item.issue_order.items():
                    loop.actor_order(role, [(name, windows[name]) for name in names])
                for source, target, distance in item.hints:
                    loop.schedule.after(source, target, distance=distance)
                if item.epilogue:
                    with loop.epilogue():
                        walk(item.epilogue)

    walk(program.items)
    if not program.work_groups:
        return kb.program(name="kernel")
    units = sight.WorkUnits(groups=tuple((gid, sight.WorkGroup(loop_trip_counts=trips)) for gid, trips in program.work_groups),
                            order=tuple(sight.WorkRun(gid, count) for gid, count in program.work_order))
    return kb.program(name="kernel", work_units=units)
