"""High-level kernel builder close to TileLang's expression style.

Users declare tensors, on-chip buffers, tile copies, GEMM/elementwise/reduce
work, loops and (rarely) explicit synchronisation.  The builder derives op
and access ids, work amounts, same-iteration data dependencies from buffer
reads/writes, buffer lifetimes from the storage/stage usage, loop-carried
chains for in-place accumulators, and actor/resource structure.  Everything
it cannot derive uniquely (issue-order lookahead between actors, extra
cross-iteration constraints) is an explicit hint; ambiguous descriptions
raise ``ContractError`` instead of being guessed.

``Actor``/``Pipeline``/``Dependency``/``Carry``/``Timing`` remain available as
the expert layer; the builder emits exactly that IR, so both paths share one
``Program``, one validation and one analysis.

Example::

    kb = KernelBuilder("main", grid={"m": tiles_m, "n": tiles_n}, threads=256)
    A = kb.tensor("A", (M, K), "bf16")
    B = kb.tensor("B", (K, N), "bf16")
    C = kb.tensor("C", (M, N), "bf16")
    A_s = kb.shared("A_s", (bm, bk), "bf16", stages=3)
    B_s = kb.shared("B_s", (bk, bn), "bf16", stages=3)
    acc = kb.fragment("acc", (bm, bn), "fp32", carried=True)
    with kb.loop("k", K // bk, stages=3):
        kb.copy(A["m", "k"], A_s)
        kb.copy(B["k", "n"], B_s)
        kb.gemm(A_s, B_s, acc)
    kb.copy(acc, C["m", "n"])
    program = kb.program()

Grid dimensions are given in TileLang ``T.Kernel`` order (first dimension
varies fastest); the builder stores them as work axes with the last axis
fastest, so ``dispatch_order='linear_block_id'`` matches ``blockIdx``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence as SeqT, Tuple, Union

from ..ops.schema import (
    DTypeSpec, ElementwiseOpSpec, GemmOpSpec, ReduceOpSpec, ReductionAxisSpec, ScalarWorkSpec,
    SemanticOpSpec, TensorOperandSpec, TensorResultSpec,
)
from .contract import (
    Actor, Buffer, BufferSlots, Carry, Compute, ContractError, Dependency, EventRef, FromWorkGroup, StorageGroup,
    BlockMapper, Launch, LaunchEdge, Load, Loop, Op, Persistent, Pipeline, Program, Projection, Sequence, Store,
    Sync, TileAccess, WorkAxis, WorkGroup, WorkRun, WorkUnits, dispatch_coordinates, done, start,
)

ON_CHIP = ("smem", "register", "tmem")


def _elements(shape: SeqT[int]) -> int:
    result = 1
    for item in shape:
        result *= int(item)
    return result


@dataclass(frozen=True)
class Tiled:
    """``tiles("k")``: this tensor dimension is walked in tiles by axis ``k``."""

    axis: str


def tiles(axis: str) -> Tiled:
    return Tiled(axis)


@dataclass(frozen=True)
class TensorRef:
    """A global tensor declared on the builder."""

    name: str
    shape: Tuple[int, ...]
    dtype: str

    @property
    def element_bytes(self) -> int:
        return DTypeSpec(self.dtype).storage_bytes

    def __getitem__(self, items: Any) -> "TensorSlice":
        """One entry per tensor dimension, as in TileLang ``K[bz, k*bn:(k+1)*bn, by, :]``.

        ``"axis"`` or ``("a", "b")`` picks one index (identity varies with the
        axes), ``tiles("axis")`` walks the dimension in tiles, ``:`` takes the
        whole dimension.  All-string shorthand ``A["m", "k"]`` means every
        dimension is tiled and is only accepted when ranks match.
        """

        if not isinstance(items, tuple):
            items = (items,)
        return TensorSlice(self, tuple(items))


@dataclass(frozen=True)
class TensorSlice:
    tensor: TensorRef
    items: Tuple[Any, ...]

    def resolve(self, buffer_shape: Tuple[int, ...], where: str) -> "ResolvedSlice":
        tensor = self.tensor
        items = self.items
        if len(items) != len(tensor.shape):
            raise ContractError("%s: %s has rank %d but the slice names %d dimensions"
                                % (where, tensor.name, len(tensor.shape), len(items)))
        if all(isinstance(item, str) for item in items):
            if len(items) != len(buffer_shape):
                raise ContractError(
                    "%s: all-string slice of %s is shorthand for tiling every dimension, but the buffer has rank %d; "
                    "use tiles(axis), axis picks and ':' explicitly" % (where, tensor.name, len(buffer_shape)))
            items = tuple(Tiled(item) for item in items)
        axes = []  # type: List[str]
        tiled = []  # type: List[Tuple[str, int, int]]
        picks = []  # type: List[Tuple[Tuple[str, ...], int]]
        tile_shape = []  # type: List[int]
        valid = []  # type: List[int]
        extents = []  # type: List[int]
        dim_axes = []  # type: List[Optional[str]]
        position = 0
        for dim, item in enumerate(items):
            extent = tensor.shape[dim]
            if isinstance(item, (str, tuple)):
                names = (item,) if isinstance(item, str) else tuple(item)
                axes.extend(names)
                picks.append((names, extent))
                continue
            if position >= len(buffer_shape):
                raise ContractError("%s: slice of %s needs more buffer dimensions than %r" % (where, tensor.name, buffer_shape))
            size = buffer_shape[position]
            if isinstance(item, Tiled):
                axes.append(item.axis)
                tiled.append((item.axis, extent, size))
                tile_shape.append(size)
                valid.append(min(size, extent))
                extents.append(extent)
                dim_axes.append(item.axis)
            elif isinstance(item, slice) and item == slice(None):
                if size != extent:
                    raise ContractError("%s: ':' on %s dim %d has extent %d but the buffer dimension is %d"
                                        % (where, tensor.name, dim, extent, size))
                tile_shape.append(size)
                valid.append(size)
                extents.append(extent)
                dim_axes.append(None)
            else:
                raise ContractError("%s: unsupported slice entry %r for %s" % (where, item, tensor.name))
            position += 1
        if position != len(buffer_shape):
            raise ContractError("%s: slice of %s covers %d buffer dimensions, buffer has %d"
                                % (where, tensor.name, position, len(buffer_shape)))
        if len(axes) != len(set(axes)):
            raise ContractError("%s: an axis appears twice in the slice of %s" % (where, tensor.name))
        return ResolvedSlice(tuple(axes), tuple(tiled), tuple(picks), tuple(tile_shape), tuple(valid),
                             tuple(extents), tuple(dim_axes))


# One entry per buffer dimension: None = fully valid, or (axis, useful fraction of
# the last tile, number of tiles along the axis).  With one tile the fraction
# applies to every execution ("static"); otherwise only at the axis' last index.
RangeEntry = Optional[Tuple[str, float, int]]
Range = Tuple[RangeEntry, ...]


@dataclass(frozen=True)
class ResolvedSlice:
    axes: Tuple[str, ...]
    tiled: Tuple[Tuple[str, int, int], ...]  # (axis, tensor extent, tile extent)
    picks: Tuple[Tuple[Tuple[str, ...], int], ...]
    tile_shape: Tuple[int, ...]
    valid_shape: Tuple[int, ...]
    tensor_extents: Tuple[int, ...] = ()
    dim_axes: Tuple[Optional[str], ...] = ()

    def tile_counts(self) -> Dict[str, int]:
        return {axis: -(-extent // size) for axis, extent, size in self.tiled}

    def range(self) -> Range:
        """Valid range of the tile per buffer dimension."""

        by_axis = {axis: (extent, size) for axis, extent, size in self.tiled}
        result = []
        for axis in self.dim_axes:
            if axis is None:
                result.append(None)
                continue
            extent, size = by_axis[axis]
            count = -(-extent // size)
            last = extent - (count - 1) * size
            result.append(None if last >= size else (axis, last / float(size), count))
        return tuple(result)


def _entries(range_: Range) -> Tuple[Tuple[str, float, int], ...]:
    """Distinct partial axes of a range (smallest fraction wins)."""

    best = {}  # type: Dict[str, Tuple[str, float, int]]
    for entry in range_:
        if entry is not None and (entry[0] not in best or entry[1] < best[entry[0]][1]):
            best[entry[0]] = entry
    return tuple(best.values())


def _narrowest(*entries: RangeEntry) -> RangeEntry:
    present = [e for e in entries if e is not None]
    return min(present, key=lambda e: e[1]) if present else None


@dataclass(frozen=True)
class BufferRef:
    """An on-chip buffer: ``smem``, ``register`` (fragment) or ``tmem``."""

    name: str
    storage: str
    shape: Tuple[int, ...]
    dtype: str
    stages: int = 1
    carried: bool = False
    execution_scope: str = "cta"

    @property
    def bytes(self) -> int:
        return _elements(self.shape) * DTypeSpec(self.dtype).storage_bytes


@dataclass
class _Stmt:
    op: Op
    reads: Tuple[str, ...]
    writes: Tuple[str, ...]
    touches: Tuple[str, ...]  # carried buffers used in place
    actor: str
    window_offset: int
    resource: Optional[str]
    partial: Tuple[Tuple[str, float, int], ...] = ()  # (axis, last-tile fraction, tile count) scaling useful work
    static_in_access: bool = False  # single-tile (static) fractions already live in TileAccess.valid_shape
    updates: Tuple[str, ...] = ()   # carried buffers this statement modifies in place (the rest of ``touches`` is read-only)


@dataclass
class _LoopNode:
    loop_id: str
    trips: Union[int, FromWorkGroup]
    stages: Optional[int]
    body: List[Any] = field(default_factory=list)
    prologue: List[Any] = field(default_factory=list)
    epilogue: List[Any] = field(default_factory=list)
    hints: List[Tuple[str, str, int, str]] = field(default_factory=list)  # (src, dst, distance, kind)
    anchor: Optional[str] = None
    actor_orders: Dict[str, Tuple[Tuple[str, int], ...]] = field(default_factory=dict)
    window_overrides: Dict[str, int] = field(default_factory=dict)


class LoopContext:
    def __init__(self, builder: "KernelBuilder", node: _LoopNode) -> None:
        self._builder = builder
        self.node = node

    def __enter__(self) -> "LoopContext":
        self._builder._stack.append(self.node.body)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._builder._stack.pop()
        return False

    def prologue(self) -> "_Section":
        return _Section(self._builder, self.node.prologue)

    def epilogue(self) -> "_Section":
        return _Section(self._builder, self.node.epilogue)

    @property
    def schedule(self) -> "ScheduleConstraints":
        """Advanced layer: constraints that pin the *source program's* schedule (barriers, warp issue order,
        window labels).  Ordinary programs only state compute and data movement and never need it."""

        return ScheduleConstraints(self)

    def after(self, source: Union[str, Op], target: Union[str, Op], *, distance: int = 0) -> None:
        """Same as ``loop.schedule.after`` (kept for existing programs)."""

        self.node.hints.append((_op_id(source), _op_id(target), distance, "done"))

    def anchor(self, op: Union[str, Op]) -> None:
        self.node.anchor = _op_id(op)

    def actor_order(self, actor: str, sequence: SeqT[Tuple[Union[str, Op], int]]) -> None:
        """Explicit issue order of one actor: ``[(op, window_offset), ...]`` over exactly its body ops.

        By default an actor issues its ops in statement order.  A source whose warp
        interleaves work of different iterations (e.g. ``PV0[k], QK0[k+1], PV1[k],
        QK1[k+1]``) states that order here; statements stay in data-flow order.
        """

        self.node.actor_orders[actor] = tuple((_op_id(op), int(offset)) for op, offset in sequence)


class ScheduleConstraints:
    """``loop.schedule``: explicit scheduling constraints of a pipelined loop.

    Everything declared here shows up by name in the lowered constraint system
    (``hint:<src>-><dst>`` dependencies, actor sequences, window offsets), in
    ``diagnostics.notes`` of the analysis, and -- when it limits the period -- on the
    critical cycle reported by ``periodic_diagnostics.explain_loop``.
    """

    def __init__(self, loop: LoopContext) -> None:
        self._loop = loop

    def after(self, source: Union[str, Op], target: Union[str, Op], *, distance: int = 0) -> None:
        """``target`` starts only after ``source`` has completed, ``distance`` iterations later (a source barrier)."""

        self._loop.after(source, target, distance=distance)

    def actor_order(self, actor: str, sequence: SeqT[Tuple[Union[str, Op], int]]) -> None:
        """Issue order of one warp/actor, ``[(op, window_offset), ...]``."""

        self._loop.actor_order(actor, sequence)

    def window(self, op: Union[str, Op], offset: int) -> None:
        """Label which iteration of ``op`` shares a steady-state window with offset-0 ops (``k + offset``)."""

        self._loop.node.window_overrides[_op_id(op)] = int(offset)

    def anchor(self, op: Union[str, Op]) -> None:
        self._loop.anchor(op)


class _Section:
    def __init__(self, builder: "KernelBuilder", items: List[Any]) -> None:
        self._builder = builder
        self._items = items

    def __enter__(self) -> "_Section":
        self._builder._stack.append(self._items)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self._builder._stack.pop()
        return False


def _broadcastable(shape: SeqT[int], target: SeqT[int]) -> bool:
    if len(shape) > len(target):
        return False
    padded = (1,) * (len(target) - len(shape)) + tuple(shape)
    return all(a == 1 or a == b for a, b in zip(padded, target))


@dataclass(frozen=True)
class MappedOperand:
    """An elementwise input whose shape neither equals nor broadcasts to the output: the mapping is explicit."""

    buffer: BufferRef
    index_map: Tuple[Tuple[str, Any], ...]


def _operand(item: Union[BufferRef, MappedOperand], output: BufferRef, where: str) -> TensorOperandSpec:
    """Identity or NumPy-style broadcast; anything else needs an explicit ``MappedOperand``."""

    if isinstance(item, MappedOperand):
        buffer = item.buffer
        return TensorOperandSpec(buffer.name, output.shape, buffer.dtype, access="affine",
                                 index_map=dict(item.index_map, declared_shape=tuple(buffer.shape)))
    buffer = item
    if tuple(buffer.shape) == tuple(output.shape):
        return TensorOperandSpec(buffer.name, buffer.shape, buffer.dtype)
    if _broadcastable(buffer.shape, output.shape):
        return TensorOperandSpec(buffer.name, buffer.shape, buffer.dtype, access="broadcast")
    raise ContractError(
        "%s: input %s%r does not broadcast to output %s%r; for a per-row state use kb.row_state(%s), for any other "
        "gather/affine access use kb.mapped(%s, {...})"
        % (where, buffer.name, tuple(buffer.shape), output.name, tuple(output.shape), buffer.name, buffer.name))


def _op_id(value: Union[str, Op]) -> str:
    return value if isinstance(value, str) else value.op_id


class KernelBuilder:
    """Concise front end that emits the typed ``Program`` contract."""

    def __init__(
        self,
        name: str,
        *,
        grid: Mapping[str, int],
        threads: int = 128,
        cluster: Tuple[int, ...] = (1, 1, 1),
        residency: Union[str, int] = "auto",
        requires_arch: Tuple[str, ...] = (),
        scheduler: Union[str, Persistent] = "static_dispatch",
        dispatch_order: Union[str, BlockMapper] = "linear_block_id",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.name = name
        self._dispatch_order = dispatch_order
        self._grid = tuple(grid.items())
        self._threads = threads
        self._cluster = tuple(cluster)
        self._residency = residency
        self._requires_arch = tuple(requires_arch)
        self._scheduler = scheduler
        self._metadata = dict(metadata or {})
        self._tensors = {}  # type: Dict[str, TensorRef]
        self._buffers = {}  # type: Dict[str, BufferRef]
        self._groups = []  # type: List[StorageGroup]
        self._inherit_stages = set()  # SMEM buffers whose slot count was left to the enclosing pipelined loop
        self._root = []  # type: List[Any]
        self._stack = [self._root]  # type: List[List[Any]]
        self._loop_ids = []  # type: List[str]
        self._loop_trips = {}  # type: Dict[str, Any]
        self._buffer_range = {}  # type: Dict[str, Range]
        self._op_ids = set()  # type: set
        self._counter = 0

    # -- declarations -------------------------------------------------------

    @property
    def work_axes(self) -> Tuple[WorkAxis, ...]:
        # TileLang grid order (x fastest) -> contract order (last axis fastest).
        return tuple(WorkAxis(name, extent) for name, extent in reversed(self._grid))

    def from_work_group(self) -> FromWorkGroup:
        return FromWorkGroup()

    def tensor(self, name: str, shape: SeqT[int], dtype: str) -> TensorRef:
        if name in self._tensors:
            raise ContractError("KernelBuilder(%s): tensor %r already declared" % (self.name, name))
        self._tensors[name] = TensorRef(name, tuple(int(x) for x in shape), DTypeSpec(dtype).name)
        return self._tensors[name]

    def _buffer(self, name: str, storage: str, shape: SeqT[int], dtype: str, stages: int, carried: bool,
                execution_scope: str) -> BufferRef:
        if name in self._buffers or name in self._tensors:
            raise ContractError("KernelBuilder(%s): buffer %r already declared" % (self.name, name))
        if carried and stages != 1:
            raise ContractError("KernelBuilder(%s): carried buffer %r cannot be multi-stage" % (self.name, name))
        ref = BufferRef(name, storage, tuple(int(x) for x in shape), DTypeSpec(dtype).name, stages, carried, execution_scope)
        self._buffers[name] = ref
        return ref

    def shared(self, name: str, shape: SeqT[int], dtype: str, *, stages: Optional[int] = None) -> BufferRef:
        """SMEM buffer.  ``stages`` = physical slot count.  ``None``: one slot, except that a buffer *written in
        the body* of ``kb.loop(..., stages=k)`` gets that loop's ``k`` slots (TileLang ``alloc_shared([k, ...])``)."""

        ref = self._buffer(name, "smem", shape, dtype, 1 if stages is None else stages, False, "cta")
        if stages is None:
            self._inherit_stages.add(name)
        return ref

    def fragment(self, name: str, shape: SeqT[int], dtype: str, *, carried: bool = False,
                 execution_scope: str = "warpgroup") -> BufferRef:
        """Register-resident fragment; ``carried=True`` marks an in-place loop accumulator."""

        return self._buffer(name, "register", shape, dtype, 1, carried, execution_scope)

    def tmem(self, name: str, shape: SeqT[int], dtype: str, *, stages: int = 1, carried: bool = False) -> BufferRef:
        return self._buffer(name, "tmem", shape, dtype, stages, carried, "cta")

    # -- structure ------------------------------------------------------------

    def _group(self, kind: str, name: str, members: SeqT[BufferRef], slots: int) -> None:
        refs = list(members)
        storages = {ref.storage for ref in refs}
        if len(refs) < 2 or len(storages) != 1 or not storages <= {"smem", "tmem"}:
            raise ContractError("KernelBuilder(%s): %s %r needs at least two buffers of one smem/tmem storage" % (self.name, kind, name))
        for ref in refs:
            if ref.carried:
                raise ContractError("KernelBuilder(%s): carried buffer %r cannot join %s %r" % (self.name, ref.name, kind, name))
            if any(ref.name in group.members for group in self._groups):
                raise ContractError("KernelBuilder(%s): buffer %r already belongs to a storage group" % (self.name, ref.name))
        self._groups.append(StorageGroup(name, kind, tuple(ref.name for ref in refs), slots))

    def alias(self, name: str, members: SeqT[BufferRef], *, slots: int = 1) -> None:
        """``members`` reuse the same physical slot one after another (e.g. P written over S).

        The slot is held from the first member's writer until the last member's
        last reader; the next iteration's first writer waits for that release.
        """

        self._group("alias", name, members, slots)

    def ring(self, name: str, members: SeqT[BufferRef], *, slots: int) -> None:
        """``members`` are allocated round-robin (in the listed order) from one FIFO ring of ``slots`` slots.

        Allocation ``n`` reuses the slot of allocation ``n - slots``: with two members
        and an even ring, K[i+slots/2] waits for the release of K[i]; with an odd ring
        the reuse crosses buffers (K[i] -> V[i+h], V[i] -> K[i+h+1]).
        """

        self._group("ring", name, members, slots)

    def loop(self, loop_id: str, trips: Union[int, FromWorkGroup], *, stages: Optional[int] = None) -> LoopContext:
        """``stages=None`` is a serial loop; ``stages=k`` is a software pipeline.

        ``k`` is the pipeline depth: SMEM buffers written in the body whose own ``stages`` was left unspecified
        get ``k`` slots; buffers with an explicit ``stages``, TMEM buffers, fragments and ring/alias groups keep
        theirs.  ``analyze`` lists the effective slot count of every buffer in ``diagnostics.notes``.
        """

        if loop_id in self._loop_ids:
            raise ContractError("KernelBuilder(%s): loop %r already declared" % (self.name, loop_id))
        self._loop_ids.append(loop_id)
        self._loop_trips[loop_id] = trips
        node = _LoopNode(loop_id, trips, stages)
        self._stack[-1].append(node)
        return LoopContext(self, node)

    # -- statements -----------------------------------------------------------

    def _unique(self, base: str) -> str:
        candidate = base
        while candidate in self._op_ids:
            self._counter += 1
            candidate = "%s_%d" % (base, self._counter)
        self._op_ids.add(candidate)
        return candidate

    def _record(self, op: Op, reads: SeqT[BufferRef], writes: SeqT[BufferRef], touches: SeqT[BufferRef],
                actor: str, window_offset: int, resource: Optional[str],
                partial: SeqT[Tuple[str, float, int]] = (), output_range: Optional[Range] = None,
                outputs: SeqT[BufferRef] = (), static_in_access: bool = False,
                updates: Optional[SeqT[BufferRef]] = None) -> Op:
        """Record a statement.

        ``partial`` scales the op's own useful work.  ``output_range`` is the
        valid range of what the op writes; it *replaces* the previous range of
        every buffer in ``outputs`` (an overwrite forgets the old contents).
        """

        # Carried buffers: those that are outputs (or explicitly updated) are modified in place, the rest only read.
        modified = [b for b in touches if b in list(outputs)] if updates is None else list(updates)
        self._stack[-1].append(_Stmt(
            op, tuple(b.name for b in reads), tuple(b.name for b in writes), tuple(b.name for b in touches),
            actor, window_offset, resource, tuple(partial), static_in_access, tuple(b.name for b in modified),
        ))
        for buffer in outputs:
            self._buffer_range[buffer.name] = (
                tuple(output_range) if output_range is not None else (None,) * len(buffer.shape)
            )
        return op

    def _range(self, buffer: BufferRef) -> Range:
        return self._buffer_range.get(buffer.name, (None,) * len(buffer.shape))

    @staticmethod
    def _split(buffers: SeqT[BufferRef]) -> Tuple[List[BufferRef], List[BufferRef]]:
        plain = [b for b in buffers if not b.carried]
        carried = [b for b in buffers if b.carried]
        return plain, carried

    def _check_axis_counts(self, resolved: ResolvedSlice, where: str) -> None:
        extents = dict(self._grid)
        for axis, count in resolved.tile_counts().items():
            declared = extents.get(axis, self._loop_trips.get(axis))
            if isinstance(declared, int) and declared != count:
                raise ContractError("%s: axis %r has %d iterations but the tensor needs %d tiles" % (where, axis, declared, count))
        for names, extent in resolved.picks:
            product = 1
            known = True
            for name in names:
                declared = extents.get(name, self._loop_trips.get(name))
                if not isinstance(declared, int):
                    known = False
                    break
                product *= declared
            if known and product != extent:
                raise ContractError("%s: pick by %s covers %d indices but the tensor dimension has %d"
                                    % (where, list(names), product, extent))

    def copy(
        self,
        src: Union[TensorSlice, BufferRef],
        dst: Union[TensorSlice, BufferRef],
        *,
        engine: Optional[str] = None,
        name: Optional[str] = None,
        actor: Optional[str] = None,
        window_offset: int = 0,
    ) -> Op:
        """Tile copy between a global tensor slice and a buffer, or between buffers.

        Tile shape, valid range and tail fractions come from the tensor shape
        and the buffer shape.  A dtype change is never folded into the move:
        buffer-to-buffer copies are lowered to the move (in the representation
        that actually travels) plus an explicit register-side ``convert``;
        tensor/buffer copies must agree on dtype.
        """

        if isinstance(src, TensorSlice) and isinstance(dst, BufferRef):
            tensor = src.tensor
            where = "copy(%s -> %s)" % (tensor.name, dst.name)
            if tensor.dtype != dst.dtype:
                raise ContractError("%s: dtype %s != %s; load into a %s buffer and cast() in registers"
                                    % (where, tensor.dtype, dst.dtype, tensor.dtype))
            resolved = src.resolve(dst.shape, where)
            self._check_axis_counts(resolved, where)
            access = self._access(tensor, resolved)
            engine = engine or ("tma" if dst.storage == "smem" else "ldst")
            op_id = self._unique(name or "load_%s" % tensor.name)
            actor = actor or "producer"
            op = Load(op_id, "global", dst.storage, engine, access=access, actor=actor,
                      writes=() if dst.carried else (dst.name,), reads=(dst.name,) if dst.carried else (),
                      metadata={"tensor_dtype": tensor.dtype})
            return self._record(op, [], [] if dst.carried else [dst], [dst] if dst.carried else [], actor,
                                window_offset, "tma" if engine == "tma" else None,
                                partial=_entries(resolved.range()), output_range=resolved.range(), outputs=[dst],
                                static_in_access=True)
        if isinstance(src, BufferRef) and isinstance(dst, TensorSlice):
            tensor = dst.tensor
            where = "copy(%s -> %s)" % (src.name, tensor.name)
            if tensor.dtype != src.dtype:
                raise ContractError("%s: dtype %s != %s; cast() into a %s fragment before the store"
                                    % (where, src.dtype, tensor.dtype, tensor.dtype))
            resolved = dst.resolve(src.shape, where)
            self._check_axis_counts(resolved, where)
            access = self._access(tensor, resolved)
            engine = engine or ("tma" if src.storage == "smem" else "ldst")
            op_id = self._unique(name or "store_%s" % tensor.name)
            actor = actor or "epilogue"
            op = Store(op_id, src.storage, "global", engine, access=access, actor=actor, reads=(src.name,),
                       metadata={"tensor_dtype": tensor.dtype})
            return self._record(op, [src] if not src.carried else [], [], [src] if src.carried else [], actor,
                                window_offset, "tma" if engine == "tma" else None,
                                partial=_entries(resolved.range()), static_in_access=True)
        if isinstance(src, BufferRef) and isinstance(dst, BufferRef):
            return self._buffer_copy(src, dst, engine or "ldst", name, actor or "compute", window_offset)
        raise ContractError("copy needs (tensor slice -> buffer), (buffer -> tensor slice) or (buffer -> buffer)")

    @staticmethod
    def _access(tensor: TensorRef, resolved: ResolvedSlice) -> TileAccess:
        """Tile access whose valid range and tensor extents come from the declared tensor shape."""

        return TileAccess(
            tensor.name, Projection(resolved.axes), resolved.tile_shape, tensor.element_bytes,
            valid_shape=None if resolved.valid_shape == resolved.tile_shape else resolved.valid_shape,
            tensor_shape=resolved.tensor_extents,
        )

    def _buffer_copy(self, src: BufferRef, dst: BufferRef, engine: str, name: Optional[str], actor: str,
                     window_offset: int) -> Op:
        where = "copy(%s -> %s)" % (src.name, dst.name)
        if _elements(src.shape) != _elements(dst.shape):
            raise ContractError("%s: element counts differ (%r vs %r); a partial or reshaping copy is not derivable"
                                % (where, src.shape, dst.shape))
        if src.storage == "register" and dst.storage == "register":
            raise ContractError("%s: register-to-register transfer is compute; use cast()" % where)
        base = name or "copy_%s_to_%s" % (src.name, dst.name)
        if src.dtype == dst.dtype:
            return self._move(src, dst, engine, self._unique(base), actor, window_offset)
        if dst.storage == "register":
            # Move the source representation, then convert in registers.
            raw = self._hidden(dst.name + "__raw", dst.shape, src.dtype)
            self._move(src, raw, engine, self._unique(base), actor, window_offset)
            return self.cast(raw, dst, name="%s_convert" % base, actor=actor, window_offset=window_offset)
        if src.storage == "register":
            # Convert in registers, then move the destination representation.
            converted = self._hidden(src.name + "__as_" + dst.dtype, src.shape, dst.dtype)
            self.cast(src, converted, name="%s_convert" % base, actor=actor, window_offset=window_offset)
            return self._move(converted, dst, engine, self._unique(base), actor, window_offset)
        raise ContractError("%s: dtype %s -> %s between %s and %s needs an explicit register stage"
                            % (where, src.dtype, dst.dtype, src.storage, dst.storage))

    def _hidden(self, name: str, shape: SeqT[int], dtype: str) -> BufferRef:
        if name in self._buffers:
            return self._buffers[name]
        return self._buffer(name, "register", shape, dtype, 1, False, "warpgroup")

    def _move(self, src: BufferRef, dst: BufferRef, engine: str, op_id: str, actor: str, window_offset: int) -> Op:
        reads = () if src.carried else (src.name,)
        writes = () if dst.carried else (dst.name,)
        touches = [b for b in (src, dst) if b.carried]
        touch_names = tuple(b.name for b in touches)
        cls = Load if dst.storage == "register" else (Store if src.storage == "register" else Load)
        op = cls(op_id, src.storage, dst.storage, engine, bytes=src.bytes, actor=actor,
                 reads=tuple(sorted(set(reads) | set(touch_names))), writes=writes)
        source_range = self._range(src)
        if len(source_range) != len(dst.shape):
            source_range = (None,) * len(dst.shape)
        return self._record(op, [] if src.carried else [src], [] if dst.carried else [dst], touches, actor,
                            window_offset, None, partial=_entries(self._range(src)), output_range=source_range,
                            outputs=[dst])

    def gemm(
        self,
        a: BufferRef,
        b: BufferRef,
        out: BufferRef,
        *,
        transpose_b: bool = False,
        name: Optional[str] = None,
        actor: str = "tensor",
        window_offset: int = 0,
        compute_dtype: Optional[str] = None,
        accumulation_dtype: str = "fp32",
        op_id: str = "gemm.dense",
    ) -> Op:
        """``out (+)= a @ b``; shapes give m/n/k. A carried ``out`` accumulates in place."""

        if len(a.shape) != 2 or len(b.shape) != 2:
            raise ContractError("gemm operands must be 2-D tiles")
        m, k = a.shape
        bk, n = (b.shape[1], b.shape[0]) if transpose_b else b.shape
        if bk != k:
            raise ContractError("gemm %s@%s: reduction extents %d and %d differ (transpose_b=%s)" % (a.name, b.name, k, bk, transpose_b))
        if tuple(out.shape) != (m, n):
            raise ContractError("gemm output %s has shape %r, expected %r" % (out.name, out.shape, (m, n)))
        spec = GemmOpSpec(op_id=op_id, m=m, n=n, k=k, a_storage_dtype=a.dtype, b_storage_dtype=b.dtype,
                          result_storage_dtype=out.dtype, compute_dtype=compute_dtype or a.dtype,
                          accumulation_dtype=accumulation_dtype)
        name = self._unique(name or "gemm_%s_%s" % (a.name, b.name))
        plain_in, carried_in = self._split([a, b])
        reads = tuple(x.name for x in plain_in)
        touches = list(carried_in)
        if out.carried:
            touches.append(out)
            writes = ()
        else:
            writes = (out.name,)
        op = Compute(name, spec, engine="tensor", actor=actor,
                     reads=tuple(sorted(set(reads) | {t.name for t in touches})), writes=writes)
        range_a, range_b = self._range(a), self._range(b)
        b_k, b_n = (range_b[1], range_b[0]) if transpose_b else (range_b[0], range_b[1])
        # Useful work shrinks along m, the reduced axis k and n; the reduced axis
        # does not limit the valid range of the (m, n) output.
        work = tuple(e for e in (range_a[0], _narrowest(range_a[1], b_k), b_n) if e is not None)
        return self._record(op, plain_in, [] if out.carried else [out], touches, actor, window_offset, "tensor",
                            partial=_entries(work), output_range=(range_a[0], b_n), outputs=[out])

    @staticmethod
    def _work(elements: int, ops: SeqT[Tuple[str, float, str]], compute_dtype: str) -> Tuple[ScalarWorkSpec, ...]:
        """One issue unit per operation; semantic FLOPs by class (design section 2.2.1).

        ``add``/``mul`` are 1 FLOP and 1 CUDA instruction unit; a fused ``fma``
        is 2 FLOPs but still 1 unit.  Separate ``mul`` + ``add`` entries are 2
        units.  SFU and conversion classes keep their own class and engine.
        """

        result = []
        for cls, count, engine in ops:
            semantic = SEMANTIC_FLOPS_PER_OP.get(cls, 1.0)
            unit = "cuda_instruction" if engine in ("cuda", "vector") else "%s_op" % engine
            result.append(ScalarWorkSpec(cls, elements * count, engine, compute_dtype, semantic, 1.0,
                                         attrs={"issue_unit": unit}))
        return tuple(result)

    def row_state(self, buffer: BufferRef) -> MappedOperand:
        """Input that holds one state row per output row (``buffer.shape[0] == output.shape[0]``), e.g. running max/sum."""

        return MappedOperand(buffer, (("kind", "row_state"), ("row_axis", 0)))

    def mapped(self, buffer: BufferRef, index_map: Mapping[str, Any]) -> MappedOperand:
        """Input read through an explicit gather/affine ``index_map`` (recorded on the operand, never inferred)."""

        if not index_map:
            raise ContractError("KernelBuilder(%s): mapped(%s) needs a non-empty index_map" % (self.name, buffer.name))
        return MappedOperand(buffer, tuple(sorted(dict(index_map).items())))

    def elementwise(
        self,
        name: str,
        inputs: SeqT[Union[BufferRef, MappedOperand]],
        output: BufferRef,
        ops: SeqT[Tuple[str, float, str]],
        *,
        updates: SeqT[BufferRef] = (),
        compute_dtype: str = "fp32",
        actor: str = "compute",
        window_offset: int = 0,
        effective_fraction: float = 1.0,
    ) -> Op:
        """Custom-work entry: ``ops`` = ((op_class, operations per output element, engine), ...).

        Inputs must equal or broadcast to the output shape; anything else is passed
        as ``kb.row_state(buf)`` / ``kb.mapped(buf, {...})``.  A ``carried`` input is
        only *read* unless it is listed in ``updates`` (state modified in place, e.g.
        the running max/sum of an online softmax); a carried ``output`` is always
        updated.  For plain arithmetic prefer ``add``/``mul``/``fma``/``exp2``/....
        """

        where = "elementwise(%s)" % name
        for item in inputs:
            if isinstance(item, MappedOperand) and dict(item.index_map).get("kind") == "row_state":
                if not item.buffer.shape or not output.shape or item.buffer.shape[0] != output.shape[0]:
                    raise ContractError("%s: row_state(%s%r) needs the same leading extent as output %s%r"
                                        % (where, item.buffer.name, tuple(item.buffer.shape), output.name, tuple(output.shape)))
        for buffer in updates:
            if not buffer.carried:
                raise ContractError("%s: updates=[%s] must be a carried buffer" % (where, buffer.name))
        operands = tuple(_operand(x, output, where) for x in inputs)
        plain_reads = [x for x in inputs if not isinstance(x, MappedOperand)]   # the operands whose range is implied
        inputs = [x.buffer if isinstance(x, MappedOperand) else x for x in inputs]
        for buffer in updates:
            if buffer not in inputs:
                inputs.append(buffer)
        elements = _elements(output.shape)
        spec = ElementwiseOpSpec(
            op_id=name,
            operands=operands or (TensorOperandSpec(output.name, output.shape, output.dtype),),
            result=TensorResultSpec(output.name, output.shape, output.dtype),
            operations=self._work(elements, ops, compute_dtype),
            compute_dtype=compute_dtype,
        )
        return self._compute(name, spec, inputs, output, actor, window_offset, effective_fraction, updates=updates,
                             range_inputs=plain_reads)

    # -- common arithmetic: work is derived, nothing to count by hand ----------------

    _UNARY = {"exp2": "sfu", "exp": "sfu", "log2": "sfu", "log": "sfu", "sqrt": "sfu", "rsqrt": "sfu",
              "neg": "cuda", "abs": "cuda"}
    _BINARY = {"add": "cuda", "sub": "cuda", "mul": "cuda", "div": "cuda", "max": "cuda", "min": "cuda"}

    def _math(self, cls: str, engine: str, inputs: SeqT[Any], output: BufferRef, name: Optional[str], kwargs: Dict[str, Any]) -> Op:
        label = name or "%s_%s" % (cls, output.name)
        return self.elementwise(label, list(inputs), output, ((cls, 1.0, engine),), **kwargs)

    def add(self, a: Any, b: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        """``out = a + b`` (one CUDA instruction per output element); ``sub``/``mul``/``div``/``maximum``/``minimum`` alike."""

        return self._math("add", "cuda", (a, b), out, name, kwargs)

    def sub(self, a: Any, b: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        return self._math("sub", "cuda", (a, b), out, name, kwargs)

    def mul(self, a: Any, b: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        return self._math("mul", "cuda", (a, b), out, name, kwargs)

    def div(self, a: Any, b: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        return self._math("div", "cuda", (a, b), out, name, kwargs)

    def maximum(self, a: Any, b: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        return self._math("max", "cuda", (a, b), out, name, kwargs)

    def minimum(self, a: Any, b: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        return self._math("min", "cuda", (a, b), out, name, kwargs)

    def fma(self, a: Any, b: Any, c: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        """``out = a * b + c``: one fused CUDA instruction (2 semantic FLOPs) per output element."""

        return self._math("fma", "cuda", (a, b, c), out, name, kwargs)

    def unary(self, function: str, a: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        """``out = function(a)`` for ``exp2``/``exp``/``log2``/``log``/``sqrt``/``rsqrt`` (SFU) and ``neg``/``abs`` (CUDA)."""

        if function not in self._UNARY:
            raise ContractError("unary(%r): supported functions are %s; use elementwise(ops=...) for custom work"
                                % (function, ", ".join(sorted(self._UNARY))))
        return self._math(function, self._UNARY[function], (a,), out, name, kwargs)

    def exp2(self, a: Any, out: BufferRef, *, name: Optional[str] = None, **kwargs: Any) -> Op:
        return self.unary("exp2", a, out, name=name, **kwargs)

    def reduce_sum(self, input: BufferRef, output: BufferRef, *, axis: int, name: Optional[str] = None, **kwargs: Any) -> Op:
        """``output = sum(input, axis)``: one add per input element."""

        return self.reduce(name or "reduce_sum_%s" % output.name, input, output, axis=axis, reducer="sum", **kwargs)

    def reduce_max(self, input: BufferRef, output: BufferRef, *, axis: int, name: Optional[str] = None, **kwargs: Any) -> Op:
        return self.reduce(name or "reduce_max_%s" % output.name, input, output, axis=axis, reducer="max",
                           ops=(("max", 1.0, "cuda"),), **kwargs)

    def reduce(
        self,
        name: str,
        input: BufferRef,
        output: BufferRef,
        *,
        axis: int,
        reducer: str = "sum",
        ops: SeqT[Tuple[str, float, str]] = (("add", 1.0, "cuda"),),
        compute_dtype: str = "fp32",
        actor: str = "compute",
        window_offset: int = 0,
    ) -> Op:
        spec = ReduceOpSpec(
            op_id=name,
            operands=(TensorOperandSpec(input.name, input.shape, input.dtype),),
            result=TensorResultSpec(output.name, output.shape, output.dtype),
            reduction_axes=(ReductionAxisSpec(axis, input.shape[axis], input.shape[axis], (input.name,)),),
            reducer=reducer,
            work=self._work(_elements(input.shape), ops, compute_dtype),
            compute_dtype=compute_dtype,
        )
        source = self._range(input)
        if len(output.shape) == len(input.shape):
            reduced = tuple(None if dim == axis else entry for dim, entry in enumerate(source))
        else:
            reduced = tuple(entry for dim, entry in enumerate(source) if dim != axis)
            if len(reduced) != len(output.shape):
                reduced = (None,) * len(output.shape)
        return self._compute(name, spec, [input], output, actor, window_offset, 1.0,
                             work_range=source, output_range=reduced)

    def cast(self, src: BufferRef, dst: BufferRef, *, name: Optional[str] = None, actor: str = "compute",
             window_offset: int = 0) -> Op:
        """Register-side dtype conversion (one ``convert`` per element, its own operation class)."""

        if _elements(src.shape) != _elements(dst.shape):
            raise ContractError("cast(%s -> %s): element counts differ" % (src.name, dst.name))
        return self.elementwise(name or "cast_%s_to_%s" % (src.name, dst.name), [src], dst,
                                (("convert", 1.0, "cuda"),), actor=actor, window_offset=window_offset)

    def compute(self, name: str, spec: SemanticOpSpec, *, reads: SeqT[BufferRef] = (), writes: SeqT[BufferRef] = (),
                engine: Optional[str] = None, actor: str = "compute", window_offset: int = 0) -> Op:
        """Expert escape hatch: a prepared semantic spec with explicit buffer usage."""

        name = self._unique(name)
        plain_in, carried_in = self._split(list(reads))
        plain_out, carried_out = self._split(list(writes))
        touches = carried_in + carried_out
        op = Compute(name, spec, engine=engine, actor=actor,
                     reads=tuple(sorted({b.name for b in plain_in} | {b.name for b in touches})),
                     writes=tuple(b.name for b in plain_out))
        resource = "tensor" if engine == "tensor" else None
        work = tuple(entry for buffer in reads for entry in self._range(buffer))
        return self._record(op, plain_in, plain_out, touches, actor, window_offset, resource,
                            partial=_entries(work), output_range=None, outputs=list(writes))

    def _compute(self, name: str, spec: SemanticOpSpec, inputs: SeqT[BufferRef], output: BufferRef, actor: str,
                 window_offset: int, effective_fraction: float, work_range: Optional[Range] = None,
                 output_range: Optional[Range] = None, updates: SeqT[BufferRef] = (),
                 range_inputs: Optional[SeqT[BufferRef]] = None) -> Op:
        name = self._unique(name)
        plain_in, carried_in = self._split(list(inputs))
        touches = list(carried_in)
        if output.carried:
            if output not in touches:
                touches.append(output)
            writes = ()
        else:
            writes = (output.name,)
        if output_range is None:
            # Elementwise: an output element is useful where every input it reads is valid.  Inputs are
            # aligned NumPy-style (trailing axes); a size-1 (broadcast) axis constrains nothing, a full axis
            # carries its valid range to the matching output axis.  An in-place output contributes the range
            # of its *old* value; an explicitly mapped *operand* carries no implied range (the mapping decides),
            # while a plain read of the same buffer in the same op still does.
            rank = len(output.shape)
            per_dim = [[] for _ in range(rank)]  # type: List[List[RangeEntry]]
            for buffer in (inputs if range_inputs is None else range_inputs):
                if len(buffer.shape) > rank:
                    continue
                offset = rank - len(buffer.shape)
                if any(extent not in (1, output.shape[offset + dim]) for dim, extent in enumerate(buffer.shape)):
                    continue
                for dim, entry in enumerate(self._range(buffer)):
                    if buffer.shape[dim] != 1 or output.shape[offset + dim] == 1:
                        per_dim[offset + dim].append(entry)
            output_range = tuple(_narrowest(*entries) for entries in per_dim)
        if work_range is None:
            work_range = output_range
        op = Compute(name, spec, engine=None, actor=actor, effective_fraction=effective_fraction,
                     reads=tuple(sorted({b.name for b in plain_in} | {b.name for b in touches})), writes=writes)
        modified = [b for b in touches if b is output or b in list(updates)]
        return self._record(op, plain_in, [] if output.carried else [output], touches, actor, window_offset, None,
                            partial=_entries(work_range), output_range=output_range, outputs=[output], updates=modified)

    def sync(self, name: str, kind: str = "barrier", *, actor: str = "compute") -> Op:
        op = Sync(self._unique(name), sync_kind=kind, actor=actor)
        return self._record(op, [], [], [], actor, 0, None)

    # -- build ------------------------------------------------------------------

    def _loop_nodes(self, items: Optional[List[Any]] = None) -> List[_LoopNode]:
        result = []  # type: List[_LoopNode]
        for item in (self._root if items is None else items):
            if isinstance(item, _LoopNode):
                result.append(item)
                for part in (item.prologue, item.body, item.epilogue):
                    result.extend(self._loop_nodes(part))
        return result

    def _slot_count(self, name: str) -> int:
        """Physical slots of a buffer, resolved from the *whole* launch (a pure function of the declarations).

        An SMEM buffer whose ``stages`` was left unspecified takes the depth of the pipelined loop(s) whose body
        writes it.  Loops that ask for different depths are ambiguous: the buffer then needs an explicit count.
        """

        ref = self._buffers[name]
        if name not in self._inherit_stages or any(name in group.members for group in self._groups):
            return ref.stages
        depths = {}  # type: Dict[int, List[str]]
        for node in self._loop_nodes():
            if node.stages and any(isinstance(item, _Stmt) and name in item.writes for item in node.body):
                depths.setdefault(node.stages, []).append(node.loop_id)
        if len(depths) > 1:
            raise ContractError(
                "KernelBuilder(%s): shared buffer %r is written in pipelined loops of different depth (%s); "
                "give it an explicit stages=<slots>"
                % (self.name, name, ", ".join("%s: stages=%d" % ("/".join(ids), depth) for depth, ids in sorted(depths.items()))))
        return next(iter(depths)) if depths else ref.stages

    def _region(self, items: List[Any]) -> Any:
        children = []
        for item in items:
            if isinstance(item, _Stmt):
                children.append(item.op)
            elif isinstance(item, _LoopNode):
                children.append(self._loop_region(item))
            else:
                raise ContractError("unexpected builder item %r" % (item,))
        if not children:
            raise ContractError("KernelBuilder(%s): empty region" % self.name)
        return children[0] if len(children) == 1 else Sequence(tuple(children))

    def _loop_region(self, node: _LoopNode) -> Loop:
        prologue = self._region(node.prologue) if node.prologue else None
        epilogue = self._region(node.epilogue) if node.epilogue else None
        if node.stages is None:
            declared = [label for label, present in (
                ("after()", node.hints), ("actor_order()", node.actor_orders), ("window()", node.window_overrides),
                ("anchor()", node.anchor)) if present]
            if declared:
                raise ContractError(
                    "loop %s is serial (stages=None): %s only apply to a pipelined loop; ops of a serial loop run in "
                    "statement order" % (node.loop_id, ", ".join(declared)))
            return Loop(node.loop_id, node.trips, body=self._region(node.body), prologue=prologue, epilogue=epilogue)
        return Loop(node.loop_id, node.trips, body=self._pipeline(node), prologue=prologue, epilogue=epilogue,
                    boundary_anchor_op=node.anchor)

    def _pipeline(self, node: _LoopNode) -> Pipeline:
        stmts = []  # type: List[_Stmt]
        for item in node.body:
            if isinstance(item, _LoopNode):
                raise ContractError(
                    "loop %s: a pipelined loop cannot contain loop %s; use a serial loop (stages=None) for the outer level"
                    % (node.loop_id, item.loop_id))
            stmts.append(item)
        if not stmts:
            raise ContractError("loop %s: pipelined body is empty" % node.loop_id)
        body_ops = {s.op.op_id for s in stmts}
        unknown = [op_id for op_id in node.window_overrides if op_id not in body_ops]
        if unknown:
            raise ContractError("loop %s: schedule.window() names ops outside the loop body: %s" % (node.loop_id, unknown))
        body_actors = {s.actor for s in stmts}
        unknown = [actor for actor in node.actor_orders if actor not in body_actors]
        if unknown:
            raise ContractError("loop %s: schedule.actor_order() names unknown actors %s (body actors: %s)"
                                % (node.loop_id, unknown, sorted(body_actors)))
        unknown = sorted({op_id for src, dst, _d, _k in node.hints for op_id in (src, dst) if op_id not in body_ops})
        if unknown:
            raise ContractError("loop %s: schedule.after() names ops outside the loop body: %s" % (node.loop_id, unknown))
        if node.anchor is not None and node.anchor not in body_ops:
            raise ContractError("loop %s: schedule.anchor() names an op outside the loop body: %s" % (node.loop_id, node.anchor))
        stmts = [replace(s, window_offset=node.window_overrides[s.op.op_id]) if s.op.op_id in node.window_overrides else s
                 for s in stmts]
        for actor_name, sequence in node.actor_orders.items():
            mine = [s for s in stmts if s.actor == actor_name]
            if sorted(op_id for op_id, _o in sequence) != sorted(s.op.op_id for s in mine):
                raise ContractError("loop %s: actor_order(%r) must list exactly the actor's body ops %s"
                                    % (node.loop_id, actor_name, [s.op.op_id for s in mine]))
            offsets = dict(sequence)
            stmts = [replace(s, window_offset=offsets[s.op.op_id]) if s.actor == actor_name else s for s in stmts]
        used = []  # type: List[str]
        for stmt in stmts:
            for name in stmt.reads + stmt.writes + stmt.touches:
                if name not in used:
                    used.append(name)
        # Lifetimes: one writer per plain smem/tmem buffer; the slot is released
        # only after *every* reader has completed.
        slots = []
        dependencies = []
        carries = []
        lifetimes = {}  # type: Dict[str, Tuple[EventRef, EventRef, int]]
        release_stmts = []  # type: List[_Stmt]
        for name in used:
            ref = self._buffers[name]
            if ref.carried:
                touchers = [s for s in stmts if name in s.touches]
                if not touchers:
                    continue
                writers = [s for s in touchers if name in s.updates]
                if not writers:
                    raise ContractError(
                        "loop %s: carried buffer %r is read by %s but never updated in the loop body; pass updates=[%s] "
                        "to the op that modifies it, or drop carried=True for a read-only buffer"
                        % (node.loop_id, name, [t.op.op_id for t in touchers], name))

                def edge(source: _Stmt, target: _Stmt, distance: int = 0) -> None:
                    dependencies.append(Dependency(done(source.op), start(target.op), distance=distance,
                                                   name="inplace:%s:%s->%s" % (name, source.op.op_id, target.op.op_id)))

                # Updates are serialised and wait for every reader of the value they overwrite; readers only
                # wait for the update they read, never for each other.
                last_writer = None  # type: Optional[_Stmt]
                readers = []  # type: List[_Stmt]          # readers of the current value
                for stmt in touchers:
                    if name in stmt.updates:
                        for reader in readers:
                            edge(reader, stmt)
                        if last_writer is not None:
                            edge(last_writer, stmt)
                        last_writer, readers = stmt, []
                    else:
                        if last_writer is not None:
                            edge(last_writer, stmt)
                        readers.append(stmt)
                first = touchers[0]
                carries.append(Carry("carry_%s" % name, done(writers[-1].op), start(first.op), 1, storage=name))
                for stmt in touchers[1:touchers.index(writers[0]) + 1]:      # others that see last iteration's value
                    if stmt is not first:
                        edge(writers[-1], stmt, 1)
                for reader in readers:                                        # read after the last update
                    if reader is not writers[0]:
                        edge(reader, writers[0], 1)
                continue
            writers = [s for s in stmts if name in s.writes]
            readers = [s for s in stmts if name in s.reads]
            if len(writers) > 1:
                raise ContractError(
                    "loop %s: buffer %r is written by %s in one pipelined iteration; declare it carried=True for an "
                    "in-place accumulator or use separate buffers"
                    % (node.loop_id, name, [w.op.op_id for w in writers]))
            if not writers or ref.storage not in ("smem", "tmem"):
                continue
            if not readers:
                raise ContractError("loop %s: buffer %r is written by %s but never read in the pipeline"
                                    % (node.loop_id, name, writers[0].op.op_id))
            if len(readers) == 1:
                lifetimes[name] = (start(writers[0].op), done(readers[0].op), self._slot_count(name))
                continue
            barrier_id = "release_%s" % name  # deterministic: building twice yields the same Program
            if barrier_id in self._op_ids:
                raise ContractError("loop %s: op id %r is reserved for the derived lifetime barrier of buffer %r"
                                    % (node.loop_id, barrier_id, name))
            barrier = Sync(barrier_id, sync_kind="barrier", actor="lifetime_%s" % name,
                           metadata={"derived": "buffer lifetime: waits for every reader of %s" % name})
            window = max(reader.window_offset for reader in readers)
            release_stmts.append(_Stmt(barrier, (), (), (), barrier.actor, window, None))
            for reader in readers:
                dependencies.append(Dependency(done(reader.op), start(barrier),
                                               name="lifetime:%s:%s" % (name, reader.op.op_id)))
            lifetimes[name] = (start(writers[0].op), done(barrier), self._slot_count(name))
        grouped = set()
        groups = []
        for group in self._groups:
            present = [member for member in group.members if member in lifetimes]
            if not present:
                continue
            if len(present) != len(group.members):
                raise ContractError("loop %s: storage group %r is only partly used in this pipeline (%s)"
                                    % (node.loop_id, group.name, present))
            groups.append(group)
            grouped.update(present)
            if group.kind == "alias":
                # one lifetime for the whole chain: first writer .. last member's last reader
                slots.append(BufferSlots(present[0], lifetimes[present[0]][0], lifetimes[present[-1]][1], group.slots))
                # inside the chain the next member overwrites the slot: it may start only after every
                # reader of the previous member has finished (an impossible order shows up as infeasible).
                for previous, following in zip(present, present[1:]):
                    dependencies.append(Dependency(lifetimes[previous][1], lifetimes[following][0],
                                                   name="alias:%s:%s->%s" % (group.name, previous, following)))
                continue
            count = len(present)
            for position, member in enumerate(present):    # FIFO ring: allocation n reuses the slot of n - slots
                target = present[(position + group.slots) % count]
                dependencies.append(Dependency(
                    lifetimes[member][1], lifetimes[target][0], distance=(position + group.slots) // count,
                    name="ring:%s:%s->%s" % (group.name, member, target)))
        for name, (acquire, release, capacity) in lifetimes.items():
            if name not in grouped:
                slots.append(BufferSlots(name, acquire, release, capacity))
        stmts = stmts + release_stmts
        # Actors in first-appearance order; resource orders from engines.
        actor_stmts = {}  # type: Dict[str, List[_Stmt]]
        for stmt in stmts:
            actor_stmts.setdefault(stmt.actor, []).append(stmt)
        actors = []
        for actor_name, items in actor_stmts.items():
            if actor_name in node.actor_orders:   # explicit issue order; data-flow chains keep statement order
                position = {op_id: index for index, (op_id, _o) in enumerate(node.actor_orders[actor_name])}
                items = sorted(items, key=lambda s: position[s.op.op_id])
            resources = {s.resource for s in items if s.resource is not None}
            resource = None
            if len(resources) == 1 and all(s.resource is not None for s in items):
                resource = resources.pop()
            elif len(resources) > 1:
                raise ContractError(
                    "loop %s: actor %r mixes fixed-order resources %s; give the ops different actors"
                    % (node.loop_id, actor_name, sorted(resources)))
            actors.append(Actor(actor_name, tuple(s.op for s in items),
                                sequence=tuple((s.op.op_id, s.window_offset) for s in items),
                                order="issue", resource=resource,
                                execution_scope="warpgroup" if actor_name not in ("producer",) else "unspecified"))
        declared = list(used)
        for stmt in self._all_stmts(node.prologue) + self._all_stmts(node.epilogue):
            for name in stmt.reads + stmt.writes + stmt.touches:
                if name not in declared:
                    declared.append(name)   # prologue/epilogue storage counts towards the work unit's capacity
        buffers = tuple(
            Buffer(self._buffers[n].name, self._buffers[n].storage, self._buffers[n].shape, self._buffers[n].dtype,
                   slots=self._slot_count(n), execution_scope=self._buffers[n].execution_scope)
            for n in declared
        )
        for src, dst, distance, kind in node.hints:
            dependencies.append(Dependency(EventRef(src, kind), start(dst), distance=distance, name="hint:%s->%s" % (src, dst)))
        return Pipeline(actors=tuple(actors), buffers=buffers, dependencies=tuple(dependencies),
                        carries=tuple(carries), buffer_slots=tuple(slots), stages=node.stages,
                        storage_groups=tuple(groups))

    # -- tails ------------------------------------------------------------------

    def _all_stmts(self, items: Optional[List[Any]] = None) -> List[_Stmt]:
        result = []  # type: List[_Stmt]
        for item in (self._root if items is None else items):
            if isinstance(item, _Stmt):
                result.append(item)
            else:
                result.extend(self._all_stmts(item.prologue))
                result.extend(self._all_stmts(item.body))
                result.extend(self._all_stmts(item.epilogue))
        return result

    def tail_fractions(self) -> Dict[str, Dict[str, float]]:
        """``op_id -> {axis: useful fraction}`` for every op whose useful work is partial.

        Single-tile axes of a tensor access are not listed for that access: their
        range already lives in ``TileAccess.valid_shape`` and must not be applied twice.
        """

        result = {}
        for stmt in self._all_stmts():
            axes = {axis: fraction for axis, fraction, count in stmt.partial
                    if not (stmt.static_in_access and count == 1)}
            if axes:
                result[stmt.op.op_id] = axes
        return result

    def _derived_work_units(self) -> Optional[WorkUnits]:
        grid_axes = [name for name, _extent in reversed(self._grid)]
        extents = dict(self._grid)
        base = {}  # type: Dict[str, float]
        grid_tails = {}  # type: Dict[str, Dict[str, float]]
        for stmt in self._all_stmts():
            op_id = stmt.op.op_id
            for axis, fraction, count in stmt.partial:
                if count == 1:
                    if not stmt.static_in_access:
                        base[op_id] = base.get(op_id, 1.0) * fraction  # every execution is partial
                elif axis in extents:
                    grid_tails.setdefault(op_id, {})[axis] = fraction
                else:
                    trips = self._loop_trips.get(axis)
                    if not isinstance(trips, int) or trips <= 0:
                        raise ContractError(
                            "KernelBuilder(%s): %s has a partial tail along loop %r whose trip count comes from the work "
                            "group; declare the useful fractions in work_units and pass tails='declared'"
                            % (self.name, op_id, axis))
                    # Every work unit runs the whole loop: average useful fraction per execution.
                    base[op_id] = base.get(op_id, 1.0) * ((trips - 1 + fraction) / float(trips))
        base = {k: v for k, v in base.items() if v < 1.0}
        if not base and not grid_tails:
            return None
        tail_axes = sorted({axis for axes in grid_tails.values() for axis in axes}, key=grid_axes.index)
        groups = {}  # type: Dict[str, WorkGroup]
        order = []  # type: List[WorkRun]
        for coordinate in dispatch_coordinates([extents[name] for name in grid_axes], self._dispatch_order):
            position = dict(zip(grid_axes, coordinate))
            at_tail = tuple(axis for axis in tail_axes if position[axis] == extents[axis] - 1)
            gid = "full" if not at_tail else "tail_" + "".join(at_tail)
            if gid not in groups:
                fractions = {}
                for op_id in set(base) | set(grid_tails):
                    value = base.get(op_id, 1.0)
                    for axis in at_tail:
                        value *= grid_tails.get(op_id, {}).get(axis, 1.0)
                    if value < 1.0:
                        fractions[op_id] = value
                groups[gid] = WorkGroup(effective_fractions=fractions)
            if order and order[-1].group_id == gid:
                order[-1] = WorkRun(gid, order[-1].count + 1)
            else:
                order.append(WorkRun(gid, 1))
        return WorkUnits(tuple(groups.items()), tuple(order))

    def launch(self, *, work_units: Optional[WorkUnits] = None, tails: str = "auto") -> Launch:
        """``tails='auto'`` derives tail work groups from tensor shapes; with explicit
        ``work_units`` the caller must fold partial-tile fractions in and pass ``tails='declared'``."""

        if not self._root:
            raise ContractError("KernelBuilder(%s): no statements" % self.name)
        if tails not in ("auto", "declared"):
            raise ContractError("tails must be 'auto' or 'declared'")
        if work_units is None:
            work_units = self._derived_work_units()
        elif tails == "auto" and self.tail_fractions():
            raise ContractError(
                "KernelBuilder(%s): tensor shapes imply partial tiles for %s but explicit work_units were given; fold the "
                "fractions into the groups and pass tails='declared'" % (self.name, sorted(self.tail_fractions())))
        region = self._region(self._root)          # resolves inherited slot counts
        used = []  # type: List[str]
        for stmt in self._all_stmts():
            for name in stmt.reads + stmt.writes + stmt.touches:
                if name not in used:
                    used.append(name)
        buffers = tuple(
            Buffer(self._buffers[n].name, self._buffers[n].storage, self._buffers[n].shape, self._buffers[n].dtype,
                   slots=self._slot_count(n), execution_scope=self._buffers[n].execution_scope)
            for n in used
        )
        groups = tuple(group for group in self._groups if any(member in used for member in group.members))
        for group in groups:
            missing = [member for member in group.members if member not in used]
            if missing:
                raise ContractError("KernelBuilder(%s): storage group %r lists unused buffers %s" % (self.name, group.name, missing))
        return Launch(
            self.name, region, self.work_axes, threads=self._threads, cluster=self._cluster,
            residency=self._residency, work_units=work_units, scheduler=self._scheduler,
            dispatch_order=self._dispatch_order, requires_arch=self._requires_arch, metadata=self._metadata,
            buffers=buffers, storage_groups=groups,
        )

    def program(self, *, work_units: Optional[WorkUnits] = None, name: Optional[str] = None, tails: str = "auto") -> Program:
        # The default program name never equals the launch name, so a target like
        # "main" is unambiguous in compare()/validation.
        return Program((self.launch(work_units=work_units, tails=tails),), name=name or "%s_program" % self.name)


SEMANTIC_FLOPS_PER_OP = {"fma": 2.0, "mul_add": 2.0}


def sequential_program(launches: SeqT[Launch], *, name: str, gaps: Mapping[Tuple[str, str], float] = ()) -> Program:
    """Chain launches in order; ``gaps`` gives explicit inter-launch gaps in seconds."""

    launches = tuple(launches)
    edges = []
    for previous, current in zip(launches, launches[1:]):
        edges.append(LaunchEdge(previous.name, current.name, gap_s=dict(gaps).get((previous.name, current.name), 0.0)))
    return Program(launches, edges=tuple(edges), name=name)


__all__ = ["BufferRef", "KernelBuilder", "LoopContext", "SEMANTIC_FLOPS_PER_OP", "TensorRef", "TensorSlice", "Tiled",
           "sequential_program", "tiles"]
