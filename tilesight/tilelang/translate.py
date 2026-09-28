"""Planned TileLang IR -> neutral tile program for TileSight's ``KernelBuilder``.

Translation only: every number written here is read from the IR, the shape
bindings or the device facts. Work is described (copies, GEMMs, element-wise
operation counts, reductions); pricing, cache traffic and scheduling are left
to TileSight. A structure outside the supported contract is refused with a
code instead of being approximated.

Main steps:

* thread predicates partition the CTA into warp-specialized roles; each role
  issues its operations in program order (one TileSight actor per role);
* shared buffers with a leading stage dimension are rings when indexed by
  ``(loop + c) % S`` and are split into separate buffers when only indexed by
  constants;
* the iteration an operation works on follows from its ring slot (reads look
  back, writes look ahead) and from data flow; a software pipeline written by
  hand (first iteration peeled, last iteration drained) is re-rolled into
  uniform iterations with window offsets after the peeled GEMMs are checked to
  be exactly the loop's GEMMs;
* role loops over the same iterations are merged into one pipelined loop.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .capture import Capture
from .errors import Refusal


def _tvm():
    from tilelang import tvm
    return tvm


def _tirx():
    return _tvm().tirx


_DTYPES = {"float16": "fp16", "bfloat16": "bf16", "float32": "fp32", "float64": "fp64", "int8": "int8",
           "int32": "int32", "uint8": "uint8", "float8_e4m3": "fp8_e4m3", "float8_e4m3fn": "fp8_e4m3",
           "float8_e5m2": "fp8_e5m2", "int64": "int64"}

# Calls that order or fence execution. The builder derives ordering from buffer use and slots;
# barriers between roles that carry no data are recorded in the notes.
_SYNC_CALLS = {
    "tl.mbarrier_wait_parity", "tl.mbarrier_expect_tx", "tl.mbarrier_arrive", "tirx.ptx_arrive_barrier",
    "tl.ptx_arrive_barrier", "tl.named_barrier_arrive", "tl.named_barrier_sync", "tirx.tvm_storage_sync",
    "tl.wait_wgmma", "tl.warpgroup_fence_operand", "tl.set_max_nreg", "tl.no_set_max_nreg", "tl.fence_proxy_async",
    "tl.sync_threads", "tl.sync_thread_partial", "tl.barrier_wait", "tl.barrier_arrive", "tirx.ptx_commit_group",
    "tirx.ptx_wait_group", "tl.cp_async_wait", "tl.cp_async_commit", "tl.warpgroup_arrive", "tl.warpgroup_commit_batch",
    "tl.warpgroup_wait",
}
_REGION_CALLS = {"tl.tileop.region", "tl.region"}  # TileLang 0.1.12, 0.1.14
_ROLE_BARRIERS = {"tl.named_barrier_arrive", "tl.named_barrier_sync", "tirx.tvm_storage_sync"}
_SFU = {"tirx.exp2": "exp2", "tirx.exp": "exp", "tirx.log": "log", "tirx.log2": "log2", "tirx.sqrt": "sqrt",
        "tirx.rsqrt": "rsqrt", "tirx.tanh": "tanh", "tirx.sin": "sin", "tirx.cos": "cos", "tirx.tan": "tan",
        "tirx.erf": "erf", "tirx.sigmoid": "sigmoid", "tl.exp2": "exp2", "tirx.exp10": "exp10", "tirx.log10": "log10",
        "tirx.log1p": "log1p", "tirx.expm1": "expm1", "tirx.atan": "atan", "tirx.atan2": "atan2", "tirx.asin": "asin",
        "tirx.acos": "acos", "tirx.sinh": "sinh", "tirx.cosh": "cosh", "tirx.asinh": "asinh", "tirx.acosh": "acosh",
        "tirx.atanh": "atanh", "tirx.pow": "pow", "tl.__exp": "exp", "tl.__exp10": "exp10", "tl.__log": "log",
        "tl.__log2": "log2", "tl.__log10": "log10", "tl.__sin": "sin", "tl.__cos": "cos", "tl.__tan": "tan",
        "tl.rsqrt": "rsqrt"}
_CUDA_CALLS = {"tirx.fabs": "abs", "tirx.if_then_else": "select", "tirx.floor": "floor", "tirx.ceil": "ceil",
               "tirx.round": "round", "tirx.nearbyint": "round", "tirx.trunc": "trunc", "tirx.fmod": "fmod",
               "tirx.bitwise_and": "bitwise", "tirx.bitwise_or": "bitwise", "tirx.bitwise_xor": "bitwise",
               "tirx.bitwise_not": "bitwise", "tirx.shift_left": "shift", "tirx.shift_right": "shift",
               "tirx.isnan": "cmp", "tirx.isinf": "cmp", "tirx.isfinite": "cmp", "tirx.copysign": "copysign",
               "tirx.popcount": "popc", "tirx.clz": "clz", "tl.min_nan": "min", "tl.max_nan": "max",
               "tirx.reinterpret": "mov", "tl.shfl_up_sync": "shfl", "tl.shfl_down_sync": "shfl",
               "tl.shfl_xor_sync": "shfl", "tl.shfl_sync": "shfl", "tirx.tvm_warp_shuffle": "shfl",
               "tirx.tvm_warp_shuffle_up": "shfl", "tirx.tvm_warp_shuffle_down": "shfl",
               "tirx.tvm_warp_shuffle_xor": "shfl"}
_LEAF_CALLS = {"tl.infinity", "tirx.infinity"}


def _dtype(dtype: Any) -> str:
    name = str(dtype)
    if name not in _DTYPES:
        raise Refusal("DTYPE_UNSUPPORTED", "dtype %s has no TileSight mapping" % name)
    return _DTYPES[name]


def _op_name(call: Any) -> str:
    op = call.op
    return str(getattr(op, "name", op))


def _elements(shape: Sequence[int]) -> int:
    result = 1
    for extent in shape:
        result *= int(extent)
    return result


# --------------------------------------------------------------------------- neutral program

@dataclass
class Tensor:
    name: str
    shape: Tuple[int, ...]
    dtype: str
    squeeze: Tuple[int, ...] = ()   # source dims of extent 1 dropped from the declaration
    source: str = ""                # global buffer this declaration views
    view: Optional[Tuple[Any, ...]] = None  # ("box", per-dim specs) | ("flat",): a re-declaration of source bytes


@dataclass
class Buf:
    name: str
    storage: str                    # smem | register
    shape: Tuple[int, ...]          # logical shape (stage dimension removed, unit dims dropped)
    dtype: str
    slots: Optional[int] = None     # ring depth
    carried: bool = False
    base: str = ""                  # source buffer (SSA versions and stage splits keep it)


@dataclass
class Slice:
    tensor: str
    items: Tuple[Any, ...]          # ("tile", axis) | ("pick", (axes...)) | ("full",)


@dataclass
class Op:
    kind: str                       # load | store | move | gemm | elementwise | reduce
    name: str
    actor: Optional[str] = None
    src: Any = None                 # buffer name or Slice
    dst: Any = None
    engine: Optional[str] = None
    inputs: Tuple[str, ...] = ()
    output: Optional[str] = None
    ops: Tuple[Tuple[str, float, str], ...] = ()
    compute_dtype: str = "fp32"
    axis: int = 0
    reducer: str = "sum"
    accumulate: bool = False        # reads its own output (gemm without clear, reduce without clear)
    transpose_b: bool = False
    row_inputs: Tuple[str, ...] = ()  # inputs holding one value per output row
    mapped_inputs: Tuple[str, ...] = ()  # inputs read through an affine map (shape differs from the output)
    ring: Optional[Tuple[str, int]] = None  # ("read"|"write", slot offset c) of a ring access inside a loop
    window: int = 0
    barrier: Optional[Tuple[str, int]] = None  # ("arrive"|"sync", named barrier id) between roles


class CtaTrips:
    """A trip count that depends on the CTA index (causal or variable-length loops)."""

    def __init__(self, expr: Any) -> None:
        self.expr = expr

    def __add__(self, other: int) -> "CtaTrips":
        return CtaTrips(_tvm().arith.Analyzer().simplify(self.expr + other))

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, CtaTrips) and str(self.expr) == str(other.expr)

    def __hash__(self) -> int:
        return hash(str(self.expr))

    def __str__(self) -> str:
        return str(self.expr)

    __repr__ = __str__


@dataclass
class LoopN:
    name: str
    trips: Any                      # int, or CtaTrips
    stages: Optional[int]
    body: List[Any] = field(default_factory=list)
    prologue: List[Any] = field(default_factory=list)
    epilogue: List[Any] = field(default_factory=list)
    issue_order: Dict[str, List[str]] = field(default_factory=dict)  # role -> op names in program order
    hints: List[Tuple[str, str, int]] = field(default_factory=list)  # (arrive op, gated op, distance)
    meta: Dict[str, Dict[str, Any]] = field(default_factory=dict)    # role -> loop start, barrier events before it


@dataclass
class Program:
    grid: "OrderedDict[str, int]"
    threads: int
    tensors: Dict[str, Tensor]
    buffers: Dict[str, Buf]
    items: List[Any]
    actors: Tuple[str, ...]
    warp_specialized: bool
    notes: List[str] = field(default_factory=list)
    bindings: Tuple[Tuple[str, int], ...] = ()
    work_groups: Tuple[Tuple[str, Tuple[Tuple[str, int], ...]], ...] = ()   # group -> per-loop trips
    work_order: Tuple[Tuple[str, int], ...] = ()                           # runs of CTAs in launch order


# --------------------------------------------------------------------------- expressions

class _Env:
    """Substitutions for shape symbols, let-bound names, scalar ``local.var`` values and
    declared input contents (``bindings`` may give the values of a small input tensor
    that steers control flow, such as sequence offsets or a block table)."""

    def __init__(self, bindings: Mapping[Any, int]) -> None:
        tvm = _tvm()
        self.analyzer = tvm.arith.Analyzer()
        self.vmap: Dict[Any, Any] = {var: tvm.tirx.IntImm("int32", value) for var, value in bindings.items()}
        self.loads: Dict[Any, Any] = {}
        self.contents: Dict[Any, Tuple[Tuple[int, ...], Any]] = {}   # buffer data -> (shape, flat values | ("fill", v))

    def _content(self, node: Any) -> Any:
        """A load of declared contents: a constant, or ``a*i + b`` when the values are affine along the index."""

        tirx = _tirx()
        shape, values = self.contents[node.buffer.data]
        indices = [self.analyzer.simplify(i) for i in node.indices]
        const = lambda v: tirx.const(v, node.buffer.dtype)
        if isinstance(values, tuple) and values[0] == "fill":
            return const(values[1])
        free = [d for d, i in enumerate(indices) if not isinstance(i, tirx.IntImm)]
        if len(free) > 1:
            return None
        strides = [1] * len(shape)
        for d in range(len(shape) - 2, -1, -1):
            strides[d] = strides[d + 1] * shape[d + 1]
        base = sum(int(i.value) * strides[d] for d, i in enumerate(indices) if d not in free)
        if not free:
            return const(values[base])
        d = free[0]
        line = [values[base + k * strides[d]] for k in range(shape[d])]
        step = line[1] - line[0] if len(line) > 1 else 0
        if any(line[k] != line[0] + k * step for k in range(len(line))) or any(isinstance(v, float) for v in line):
            return None
        index = indices[d]
        return self.analyzer.simplify(tirx.Cast(node.buffer.dtype, index * step + line[0])
                                      if str(index.dtype) != str(node.buffer.dtype) else index * step + line[0])

    def subst(self, expr: Any) -> Any:
        tirx = _tirx()

        def post(node):
            if isinstance(node, tirx.BufferLoad):
                if node.buffer.data in self.loads:
                    return self.loads[node.buffer.data]
                if node.buffer.data in self.contents:
                    return self._content(node)
            return None

        for _ in range(8):  # let chains and loop counters may refer to each other
            new = expr
            if self.loads or self.contents:
                new = tirx.stmt_functor.ir_transform(tirx.Evaluate(new), None, post, ["tirx.BufferLoad"]).value
            if self.vmap:
                new = tirx.stmt_functor.substitute(new, self.vmap)
            if new.same_as(expr):
                break
            expr = new
        return self.analyzer.simplify(expr)

    def const(self, expr: Any, what: str) -> int:
        tirx = _tirx()
        if isinstance(expr, int):
            return expr
        value = self.subst(expr)
        if isinstance(value, tirx.IntImm):
            return int(value.value)
        raise Refusal("UNBOUND_SYMBOL", "%s is not constant: %s (bind the symbols it uses)" % (what, value))


def _affine(expr: Any, variables: Sequence[Any]) -> Optional[Tuple[Dict[Any, int], int]]:
    """``expr`` as sum(coeff * var) + const over ``variables``; None if not affine."""

    tirx = _tirx()
    if isinstance(expr, tirx.IntImm):
        return {}, int(expr.value)
    if isinstance(expr, tirx.Var):
        for var in variables:
            if var.same_as(expr):
                return {var: 1}, 0
        return None
    if isinstance(expr, (tirx.Add, tirx.Sub)):
        left, right = _affine(expr.a, variables), _affine(expr.b, variables)
        if left is None or right is None:
            return None
        sign = 1 if isinstance(expr, tirx.Add) else -1
        terms = dict(left[0])
        for var, coeff in right[0].items():
            terms[var] = terms.get(var, 0) + sign * coeff
        return {v: c for v, c in terms.items() if c}, left[1] + sign * right[1]
    if isinstance(expr, tirx.Mul):
        for a, b in ((expr.a, expr.b), (expr.b, expr.a)):
            if isinstance(b, tirx.IntImm):
                inner = _affine(a, variables)
                if inner is None:
                    return None
                k = int(b.value)
                return {v: c * k for v, c in inner[0].items()}, inner[1] * k
        return None
    return None


def _count_ops(value: Any, env: _Env) -> Tuple[Tuple[Tuple[str, float, str], ...], str]:
    """Per-element work of ``value``: unique sub-expressions, multiply-add fused.

    Index arithmetic inside buffer loads is address computation, not data work.
    Identical pure sub-expressions are counted once, as the compiler evaluates
    them once; a multiply whose only use is an add or subtract fuses into an FMA.
    """

    tirx = _tirx()
    tvm = _tvm()
    nodes: List[Tuple[Any, str, str, Tuple[int, ...]]] = []
    index: Dict[int, List[int]] = {}

    def visit(expr: Any) -> Optional[int]:
        if isinstance(expr, (tirx.IntImm, tirx.FloatImm, tirx.StringImm, tirx.Var, tirx.BufferLoad)):
            return None
        if isinstance(expr, tirx.Call) and _op_name(expr) in _LEAF_CALLS:
            return None
        key = int(tvm.ir.structural_hash(expr))
        for candidate in index.get(key, ()):
            if tvm.ir.structural_equal(nodes[candidate][0], expr):
                return candidate
        if isinstance(expr, tirx.Cast):
            cls, engine, children = "convert", "cuda", [expr.value]
        elif isinstance(expr, (tirx.Add, tirx.Sub, tirx.Mul, tirx.Div, tirx.Min, tirx.Max, tirx.FloorDiv,
                               tirx.FloorMod, tirx.Mod)):
            cls = {tirx.Add: "add", tirx.Sub: "sub", tirx.Mul: "mul", tirx.Div: "div", tirx.Min: "min",
                   tirx.Max: "max", tirx.FloorDiv: "div", tirx.FloorMod: "mod", tirx.Mod: "mod"}[type(expr)]
            engine, children = "cuda", [expr.a, expr.b]
        elif isinstance(expr, (tirx.LT, tirx.LE, tirx.GT, tirx.GE, tirx.EQ, tirx.NE)):
            cls, engine, children = "cmp", "cuda", [expr.a, expr.b]
        elif isinstance(expr, (tirx.And, tirx.Or)):
            cls, engine, children = "logic", "cuda", [expr.a, expr.b]
        elif isinstance(expr, tirx.Not):
            cls, engine, children = "logic", "cuda", [expr.a]
        elif isinstance(expr, tirx.Select):
            cls, engine, children = "select", "cuda", [expr.condition, expr.true_value, expr.false_value]
        elif isinstance(expr, tirx.Call):
            name = _op_name(expr)
            if name in _SFU:
                cls, engine = _SFU[name], "sfu"
            elif name in _CUDA_CALLS:
                cls, engine = _CUDA_CALLS[name], "cuda"
            else:
                raise Refusal("MATH_CALL_UNSUPPORTED", "no work class for %s" % name)
            children = list(expr.args)
        else:
            raise Refusal("EXPRESSION_UNSUPPORTED", "no work class for %s" % type(expr).__name__)
        child_ids = tuple(c for c in (visit(child) for child in children) if c is not None)
        nodes.append((expr, cls, engine, child_ids))
        index.setdefault(key, []).append(len(nodes) - 1)
        return len(nodes) - 1

    visit(env.subst(value))
    uses: Dict[int, int] = {}
    for _expr, _cls, _engine, children in nodes:
        for child in children:
            uses[child] = uses.get(child, 0) + 1
    fused = set()
    counts: Dict[Tuple[str, str], float] = {}
    for _expr, cls, engine, children in nodes:
        if cls in ("add", "sub"):
            muls = [c for c in children if nodes[c][1] == "mul" and uses.get(c) == 1 and c not in fused]
            if muls:
                fused.add(muls[0])
                cls = "fma"
        counts[(cls, engine)] = counts.get((cls, engine), 0.0) + 1.0
    if fused:
        counts[("mul", "cuda")] -= float(len(fused))
    float_dtypes = [str(e.dtype) for e, _c, _g, _k in nodes if str(e.dtype).startswith(("float", "bfloat"))]
    compute = _dtype(max(set(float_dtypes), key=float_dtypes.count)) if float_dtypes else "fp32"
    ops = tuple((cls, count, engine) for (cls, engine), count in sorted(counts.items()) if count > 0)
    return ops, compute


# --------------------------------------------------------------------------- walker

@dataclass
class _Region:
    buffer: Any
    mins: Tuple[Any, ...]
    extents: Tuple[int, ...]


class _Walker:
    def __init__(self, func: Any, capture: Capture, bindings: Mapping[str, Any]) -> None:
        self.tirx = _tirx()
        self.func = func
        self.capture = capture
        contents = {name: value for name, value in bindings.items() if not isinstance(value, int)}
        bindings = {name: value for name, value in bindings.items() if isinstance(value, int)}
        free = self._free_vars(func)
        unbound = sorted(name for name in free if name not in bindings)
        if unbound:
            raise Refusal("UNBOUND_SYMBOL", "bind the shape symbols %s" % ", ".join(unbound))
        scalars = {p.name: p for p in func.params if p not in func.buffer_map and str(p.dtype).startswith("int")
                   and p.name in bindings and p.name not in free}
        free.update(scalars)
        self.env = _Env({var: int(bindings[var.name]) for var in free.values()})
        self.bindings = tuple(sorted((name, int(bindings[name])) for name in free))
        for param in func.params:
            buffer = func.buffer_map.get(param)
            if buffer is None or buffer.name not in contents:
                continue
            shape = tuple(self.env.const(d, "shape of %s" % buffer.name) for d in buffer.shape)
            value = contents[buffer.name]
            if isinstance(value, Mapping) and set(value) == {"fill"}:
                self.env.contents[buffer.data] = (shape, ("fill", value["fill"]))
            elif isinstance(value, (list, tuple)) and len(value) == _elements(shape):
                self.env.contents[buffer.data] = (shape, tuple(value))
            else:
                raise Refusal("BINDING_INVALID", "contents of %s must be %d values or {'fill': v}"
                              % (buffer.name, _elements(shape)))
            self.bindings += (("%s[...]" % buffer.name, "declared contents"),)
        self.grid: List[Tuple[Any, str, int]] = []
        self.grid_tags: Dict[Any, str] = {}
        self.splits: Dict[str, Tuple[str, str, int]] = {}
        self.thread_var: Any = None
        self.thread_extent: Optional[int] = None
        self.tensors: Dict[str, Tensor] = {}
        self.views: Dict[str, List[Any]] = {}       # tensor -> boxes read through view declarations
        self.whole: set = set()                      # tensors read through their own declaration
        self.loop_extents: Dict[Any, int] = {}
        self.bufs: Dict[str, Buf] = {}
        self.loop_vars: Dict[Any, str] = {}
        self.loop_stack: List[Any] = []
        self.ws = False
        self.notes: List[str] = []
        self.counter = 0
        self.split_vars: Dict[str, Any] = {}
        self.thread_select: Optional[Tuple[int, int, int]] = None   # threads chosen by a predicate: first, step, count
        self.lane_origin: Dict[Any, Tuple[Any, int, str]] = {}      # split lane -> the lane it was split from
        self.data_vars = self._data_vars(func)
        self.role_barriers = 0
        self.names: Dict[Any, str] = {}
        self.stage_kind = self._prescan(func)
        for param in func.params:
            buffer = func.buffer_map.get(param)
            if buffer is not None:
                self._declared_tensor(buffer)

    # -- discovery ---------------------------------------------------------------

    def _free_vars(self, func: Any) -> Dict[str, Any]:
        tirx = self.tirx
        found: Dict[str, Any] = {}

        def visit(node):
            if isinstance(node, tirx.Var) and str(node.dtype).startswith("int"):
                found.setdefault(node.name, node)

        for buffer in func.buffer_map.values():
            for dim in list(buffer.shape) + list(buffer.strides):
                if not isinstance(dim, int):
                    tirx.stmt_functor.post_order_visit(tirx.Evaluate(dim), visit)
        return found

    def _prescan(self, func: Any) -> Dict[str, str]:
        """Shared buffers with a stage dimension: ``ring`` (some loop-indexed slot) or ``split`` (constants only)."""

        tirx = self.tirx
        seen: Dict[str, List[bool]] = {}
        whole: set = set()

        def visit(node):
            if not (isinstance(node, tirx.Call) and _op_name(node) in _REGION_CALLS):
                return
            buffer = node.args[0].buffer
            if not str(buffer.scope()).startswith("shared") or len(buffer.shape) < 2:
                return
            leading = node.args[2]
            if isinstance(leading, tirx.IntImm) and int(leading.value) == 1 and not (
                    isinstance(buffer.shape[0], tirx.IntImm) and int(buffer.shape[0]) == 1):
                seen.setdefault(self._name(buffer), []).append(isinstance(node.args[0].indices[0], tirx.IntImm))
            else:
                whole.add(self._name(buffer))

        tirx.stmt_functor.post_order_visit(func.body, visit)
        kinds: Dict[str, str] = {}
        for name, constant in seen.items():
            if name in whole:
                continue            # also accessed across its leading dimension: data, not stages
            kinds[name] = "split" if all(constant) else "ring"
        return kinds

    def _declared_tensor(self, buffer: Any) -> Tensor:
        name = buffer.name
        if name not in self.tensors:
            shape = tuple(self.env.const(dim, "shape of %s" % name) for dim in buffer.shape)
            squeeze = tuple(d for d, extent in enumerate(shape) if extent == 1 and len(shape) > 1)
            kept = tuple(extent for d, extent in enumerate(shape) if d not in squeeze)
            self.tensors[name] = Tensor(name, kept, _dtype(buffer.dtype), squeeze, name)
        return self.tensors[name]

    def _name(self, buffer: Any) -> str:
        """Unique name per buffer object: TVMScript prints same-named buffers as ``x``, ``x_1``, ..."""

        name = self.names.get(buffer.data)
        if name is None:
            taken = set(self.names.values())
            name, n = buffer.name, 0
            while name in taken:
                n += 1
                name = "%s_%d" % (buffer.name, n)
            self.names[buffer.data] = name
        return name

    def _scope(self, buffer: Any) -> str:
        return str(buffer.scope()) if hasattr(buffer, "scope") else "global"

    def _buf(self, buffer: Any, part: Optional[int] = None) -> Buf:
        """Logical on-chip buffer; a ring drops its stage dimension, a split names one stage."""

        scope = self._scope(buffer)
        storage = "smem" if scope.startswith("shared") else "register"
        shape = tuple(self.env.const(dim, "shape of %s" % buffer.name) for dim in buffer.shape)
        unique = self._name(buffer)
        kind = self.stage_kind.get(unique)
        slots, name = None, unique
        if kind == "ring":
            slots, shape = shape[0], shape[1:]
        elif kind == "split":
            if part is None:
                raise Refusal("STAGE_VIEW_UNSUPPORTED", "%s is used without a stage index" % unique)
            name, shape = "%s@%d" % (unique, part), shape[1:]
        shape = tuple(extent for extent in shape if extent != 1) or (1,)
        existing = self.bufs.get(name)
        if existing is not None:
            if existing.shape != shape:
                raise Refusal("BUFFER_VIEW_CONFLICT", "%s is used as %r and %r" % (name, existing.shape, shape))
            return existing
        record = Buf(name, storage, shape, _dtype(buffer.dtype), slots, base=unique)
        self.bufs[name] = record
        return record

    def _unique(self, base: str) -> str:
        self.counter += 1
        return "%s_%d" % (base, self.counter)

    # -- regions and indices -----------------------------------------------------

    def _region(self, expr: Any) -> _Region:
        tirx = self.tirx
        if isinstance(expr, tirx.Call) and _op_name(expr) in _REGION_CALLS:
            load = expr.args[0]
            extents = tuple(self.env.const(e, "region extent") for e in expr.args[2:])
            return _Region(load.buffer, tuple(load.indices), extents)
        if isinstance(expr, tirx.BufferLoad):
            return _Region(expr.buffer, tuple(expr.indices), tuple(1 for _ in expr.indices))
        raise Refusal("REGION_UNSUPPORTED", "operand %s is not a buffer region" % type(expr).__name__)

    def _axis_vars(self) -> List[Any]:
        return [var for var, _name, _extent in self.grid] + list(self.loop_vars) + list(self.split_vars.values())

    def _axis_name(self, var: Any) -> str:
        for gvar, name, _extent in self.grid:
            if gvar.same_as(var):
                return name
        for lvar, name in self.loop_vars.items():
            if lvar.same_as(var):
                return name
        for name, svar in self.split_vars.items():
            if svar.same_as(var):
                return name
        raise Refusal("INDEX_UNSUPPORTED", "index variable %s is neither a grid nor a loop axis" % var)

    def _grid_entry(self, var: Any) -> Optional[Tuple[Any, str, int]]:
        for entry in self.grid:
            if entry[0].same_as(var):
                return entry
        return None

    def _split(self, name: str, factor: int) -> Tuple[str, str]:
        existing = self.splits.get(name)
        if existing is not None:
            if existing[2] != factor:
                raise Refusal("INDEX_UNSUPPORTED", "grid axis %s is split by %d and %d" % (name, existing[2], factor))
            return existing[0], existing[1]
        hi, lo = "%s_div%d" % (name, factor), "%s_mod%d" % (name, factor)
        self.splits[name] = (hi, lo, factor)
        return hi, lo

    def _split_axes(self, expr: Any) -> Any:
        """``by // c`` and ``by % c`` of a grid axis become the two axes of its split."""

        tirx = self.tirx
        problems: List[Refusal] = []

        def post(node):
            if isinstance(node, (tirx.FloorDiv, tirx.FloorMod)) and isinstance(node.b, tirx.IntImm) \
                    and isinstance(node.a, tirx.Var) and self._grid_entry(node.a) is not None:
                _var, name, extent = self._grid_entry(node.a)
                factor = int(node.b.value)
                try:
                    if factor <= 1 or extent % factor:
                        raise Refusal("INDEX_UNSUPPORTED", "%d does not divide grid axis %s" % (factor, name))
                    hi, lo = self._split(name, factor)
                except Refusal as refusal:       # raising inside a TVM callback would surface as RuntimeError
                    problems.append(refusal)
                    return None
                key = hi if isinstance(node, tirx.FloorDiv) else lo
                if key not in self.split_vars:
                    self.split_vars[key] = tirx.Var(key, node.a.dtype)
                return self.split_vars[key]
            return None

        result = tirx.stmt_functor.ir_transform(tirx.Evaluate(expr), None, post,
                                                ["tirx.FloorDiv", "tirx.FloorMod"]).value
        if problems:
            raise problems[0]
        return result

    def _view(self, base: Tensor, box: List[Optional[Tuple[str, int, int]]], what: str) -> str:
        """Re-declaration of part of ``base``, the same bytes re-indexed. ``box`` has one entry per source
        dimension: None (whole), ``("range", start, length)`` or ``("blocks", m, h)`` (block ``h`` of every
        ``m`` equal blocks). The cache model sees each view as its own tensor, so views of one tensor
        must be identical or disjoint, and the tensor is not also read whole."""

        source = self._source_shape(base)
        box = tuple(box)
        boxes = self.views.setdefault(base.name, [])
        if box not in boxes:
            if base.name in self.whole or "flat" in boxes or any(
                    not any(_disjoint(a, b, self.env.const(source[d], "shape")) for d, (a, b) in enumerate(zip(box, other)))
                    for other in boxes if other != "flat"):
                self._overlap(base.name)
            boxes.append(box)
        parts, shape, squeeze = [], [], list(base.squeeze)
        for dim, extent in enumerate(source):
            spec = box[dim] if dim < len(box) else None
            if dim in base.squeeze:
                continue
            kept = dim - sum(1 for d in base.squeeze if d < dim)
            full = self.env.const(extent, "shape")
            if spec is None:
                shape.append(full)
            elif spec[0] == "blocks":
                parts.append("%d.%d" % (kept, spec[2]))
                shape.append(full // spec[1])
            elif spec[0] == "cell":
                parts.append("%d[%d]" % (kept, spec[1]))
                shape.append(1)
            elif spec[2] == 1:
                parts.append("%d[%d]" % (kept, spec[1]))
                squeeze.append(dim)
            else:
                parts.append("%d[%d:%d]" % (kept, spec[1], spec[1] + spec[2]))
                shape.append(spec[2])
        name = "%s@%s" % (base.name, ",".join(parts))
        if name not in self.tensors:
            self.tensors[name] = Tensor(name, tuple(shape) or (1,), base.dtype, tuple(sorted(squeeze)), base.name,
                                        ("box", box))
        return name

    def _flat(self, base: Tensor, what: str) -> Tensor:
        """``base`` as one dimension of all its elements (row-major): the same bytes, re-indexed."""

        boxes = self.views.setdefault(base.name, [])
        if "flat" not in boxes:
            if base.name in self.whole or boxes:
                self._overlap(base.name)
            boxes.append("flat")
        name = base.name + "@flat"
        if name not in self.tensors:
            self.tensors[name] = Tensor(name, (_elements(base.shape),), base.dtype, (), base.name, ("flat",))
        return self.tensors[name]

    def _pick_range(self, index: Any, what: str) -> Tuple[Tuple[str, ...], int, int]:
        """Axes of a single-element index and the contiguous index range they cover (start, length)."""

        index = self._split_axes(self.env.subst(index))
        terms = _affine(index, self._axis_vars())
        if terms is None or not terms[0]:
            raise Refusal("INDEX_UNSUPPORTED", "%s: index %s is not an axis pick" % (what, index))
        extents = []
        for var, coeff in terms[0].items():
            extent = self._axis_extent(var)
            if extent is None:
                raise Refusal("INDEX_UNSUPPORTED", "%s: pick by %s, whose extent varies by CTA" % (what, var))
            extents.append((coeff, extent, var))
        extents.sort(key=lambda e: e[0])
        stride = 1
        for coeff, extent, _var in extents:
            if coeff != stride:
                raise Refusal("INDEX_UNSUPPORTED", "%s: index %s is not a contiguous pick" % (what, index))
            stride *= extent
        axes: List[str] = []
        for _coeff, _extent, var in reversed(extents):
            name = self._axis_name(var)
            split = self.splits.get(name)
            axes.extend((split[0], split[1]) if split else (name,))
        return tuple(axes), terms[1], stride

    def _axis_extent(self, var: Any) -> Optional[int]:
        entry = self._grid_entry(var)
        if entry is not None:
            return entry[2]
        for name, svar in self.split_vars.items():
            if svar.same_as(var):
                for hi, lo, factor in self.splits.values():
                    if name == lo:
                        return factor
                    if name == hi:
                        return next(e for _v, n, e in self.grid if self.splits.get(n, (None,))[0] == hi) // factor
        for lvar, extent in self.loop_extents.items():
            if lvar.same_as(var):
                return extent
        return None                                  # a CTA-dependent loop: its tiles follow the work groups

    def _source_shape(self, tensor: Tensor) -> Tuple[int, ...]:
        shape = list(tensor.shape)
        for d in tensor.squeeze:
            shape.insert(d, 1)
        return tuple(shape)

    def _whole(self, base: Tensor, what: str) -> None:
        if self.views.get(base.name) and base.name not in self.whole:
            self._overlap(base.name)
        self.whole.add(base.name)

    def _overlap(self, tensor: str) -> None:
        note = ("%s is read through overlapping declarations (whole, views or flat); the cache model treats them as "
                "separate tensors, so reuse between them is not modeled and DRAM traffic may be overestimated" % tensor)
        if note not in self.notes:
            self.notes.append(note)

    def _slice(self, region: _Region, tile_shape: Tuple[int, ...], what: str) -> Slice:
        """Global region -> tiled / picked / full dimensions of a tensor declaration.

        A region that covers block ``h`` of ``m`` equal blocks of each tile step
        (``start = c*axis + h*e`` with ``c = m*e``), or one fixed block
        (``start = h*e``), reads its own declaration of that block, a bijective
        re-indexing of the same bytes. A tile walked backwards
        (``start = (n-1)*e - e*axis``) covers the same tiles in reverse order.
        """

        base = self._declared_tensor(region.buffer)
        items: List[Any] = []
        tiled: List[int] = []
        box: List[Optional[Tuple[str, int, int]]] = [None] * len(region.mins)
        for dim, (start, extent) in enumerate(zip(region.mins, region.extents)):
            if dim in base.squeeze:
                if extent != 1 or self.env.const(start, "index") != 0:
                    raise Refusal("INDEX_UNSUPPORTED", "%s: unit dimension %d of %s is not at 0" % (what, dim, base.name))
                continue
            full = self.env.const(region.buffer.shape[dim], "shape")
            start = self._split_axes(self.env.subst(start))
            if extent == full and isinstance(start, self.tirx.IntImm) and int(start.value) == 0:
                items.append(("full",))
                tiled.append(extent)
                continue
            if extent == 1:
                axes, first, count = self._pick_range(start, what)
                if (first, count) != (0, full):
                    if first < 0 or first + count > full:
                        raise Refusal("INDEX_UNSUPPORTED", "%s: %s[dim %d] picks %d of %d at %d" % (what, base.name, dim, count, full, first))
                    box[dim] = ("range", first, count)
                items.append(("pick", axes))
                continue
            terms = _affine(start, self._axis_vars())
            if terms is not None and not terms[0] and 0 <= terms[1] and terms[1] + extent <= full:
                box[dim] = ("range", terms[1], extent)                  # one fixed range
                items.append(("full",))
                tiled.append(extent)
                continue
            if terms is None or len(terms[0]) != 1:
                raise Refusal("INDEX_UNSUPPORTED", "%s: %s[dim %d] starts at %s, not a tile" % (what, base.name, dim, start))
            (var, coeff), = terms[0].items()
            offset = terms[1]
            if coeff == extent:
                box[dim] = _tile_cover(offset, extent, self._axis_extent(var), full)
                items.append(("tile", self._axis_name(var)))
            elif coeff == -extent and offset == full - extent:
                items.append(("tile", self._axis_name(var)))
                note = "%s walks %s dim %d backwards; the same tiles are modeled in forward order" % (what, base.name, dim)
                if note not in self.notes:
                    self.notes.append(note)
            elif coeff % extent == 0 and offset % extent == 0 and 0 <= offset < coeff:
                box[dim] = ("blocks", coeff // extent, offset // extent)
                items.append(("tile", self._axis_name(var)))
            else:
                raise Refusal("INDEX_UNSUPPORTED", "%s: %s[dim %d] steps %d with tile %d at %d"
                              % (what, base.name, dim, coeff, extent, offset))
            tiled.append(extent)
        if tuple(e for e in tiled if e != 1) != tuple(e for e in tile_shape if e != 1):
            raise Refusal("INDEX_UNSUPPORTED", "%s: global tile %r does not match buffer %r" % (what, tuple(tiled), tile_shape))
        if not any(box):
            self._whole(base, what)
            return Slice(base.name, tuple(items))
        return Slice(self._view(base, box, what), tuple(items))

    # -- statements ----------------------------------------------------------------

    def walk(self) -> List[Tuple[Optional[Tuple[int, int]], Any]]:
        out: List[Tuple[Optional[Tuple[int, int]], Any]] = []
        self._stmt(self.func.body, None, out)
        return out

    def _thread_interval(self, cond: Any) -> Optional[Tuple[int, int]]:
        tirx = self.tirx
        if self.thread_var is None:
            return None
        lo, hi = 0, self.thread_extent

        def bound(expr: Any) -> Optional[Tuple[int, int]]:
            if isinstance(expr, tirx.And):
                a, b = bound(expr.a), bound(expr.b)
                if a is None or b is None:
                    return None
                return max(a[0], b[0]), min(a[1], b[1])
            if isinstance(expr, (tirx.LT, tirx.LE, tirx.GE, tirx.GT)):
                a, b = expr.a, expr.b
                if isinstance(a, tirx.Var) and a.same_as(self.thread_var) and isinstance(b, tirx.IntImm):
                    v = int(b.value)
                    return {tirx.LT: (lo, v), tirx.LE: (lo, v + 1), tirx.GE: (v, hi), tirx.GT: (v + 1, hi)}[type(expr)]
                if isinstance(b, tirx.Var) and b.same_as(self.thread_var) and isinstance(a, tirx.IntImm):
                    v = int(a.value)
                    return {tirx.LT: (v + 1, hi), tirx.LE: (v, hi), tirx.GE: (lo, v + 1), tirx.GT: (lo, v)}[type(expr)]
            return None

        return bound(cond)

    def _stmt(self, stmt: Any, actor: Optional[Tuple[int, int]], out: List[Any]) -> None:
        tirx = self.tirx
        if isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                self._stmt(child, actor, out)
        elif isinstance(stmt, tirx.SBlockRealize):
            self._stmt(stmt.block, actor, out)
        elif isinstance(stmt, tirx.SBlock):
            self._stmt(stmt.body, actor, out)
        elif type(stmt).__name__ in ("AllocBuffer", "DeclBuffer", "AssertStmt"):
            if getattr(stmt, "body", None) is not None:
                self._stmt(stmt.body, actor, out)
        elif isinstance(stmt, tirx.AttrStmt):
            self._attr(stmt, actor, out)
        elif isinstance(stmt, tirx.For):
            self._for(stmt, actor, out)
        elif isinstance(stmt, tirx.IfThenElse):
            self._if(stmt, actor, out)
        elif type(stmt).__name__ in ("LetStmt", "Bind"):
            self.env.vmap[stmt.var] = stmt.value
            if getattr(stmt, "body", None) is not None:
                self._stmt(stmt.body, actor, out)
        elif isinstance(stmt, tirx.Evaluate):
            self._evaluate(stmt.value, actor, out)
        elif isinstance(stmt, tirx.BufferStore):
            self._store(stmt, actor, out)
        else:
            raise Refusal("STATEMENT_UNSUPPORTED", "no translation for %s" % type(stmt).__name__)

    def _bind_launch(self, tag: str, var: Any, extent: Any) -> None:
        if tag.startswith(("blockIdx", "threadIdx")):
            bound = self.env.const(extent, "launch extent %s" % tag)
            self.env.analyzer.bind(var, _tvm().ir.Range.from_min_extent(0, bound))
        if tag.startswith("blockIdx"):
            value = self.env.const(extent, "grid extent %s" % tag)
            name = var.name
            if any(n == name for _v, n, _e in self.grid):
                name = "%s_%s" % (name, tag[-1])
            self.grid.append((var, name, value))
            self.grid_tags[var] = tag
        elif tag == "threadIdx.x":
            self.thread_var = var
            self.thread_extent = self.env.const(extent, "thread extent")

    def _attr(self, stmt: Any, actor: Any, out: List[Any]) -> None:
        key = str(stmt.attr_key)
        if key == "thread_extent":
            self._bind_launch(str(stmt.node.thread_tag), stmt.node.var, stmt.value)
        elif key == "kWarpSpecializationScope":
            self.ws = True
        self._stmt(stmt.body, actor, out)

    def _for(self, loop: Any, actor: Any, out: List[Any]) -> None:
        tirx = self.tirx
        if loop.kind == tirx.ForKind.THREAD_BINDING:
            self._bind_launch(str(loop.thread_binding.thread_tag), loop.loop_var, loop.extent)
            self._stmt(loop.body, actor, out)
            return
        if self._element_nest(loop):
            self._parallel(loop, actor, out)
            return
        start = self.env.const(loop.min, "loop start of %s" % loop.loop_var.name)
        extent = self._trips(loop)
        if isinstance(extent, int) and extent <= 0:
            return
        self.env.analyzer.bind(loop.loop_var, _tvm().ir.Range.from_min_extent(
            start, extent if isinstance(extent, int) else extent.expr))
        if self._slot_select(loop, actor, out):
            return
        annotations = {str(k): v for k, v in (loop.annotations or {}).items()}
        stages = annotations.get("tl_pipelined_num_stages")
        name = loop.loop_var.name
        if name in self.loop_vars.values():
            name = "%s_%d" % (name, self.counter + 1)
        self.loop_vars[loop.loop_var] = name
        if isinstance(extent, int):
            self.loop_extents[loop.loop_var] = extent
        self.loop_stack.append(loop.loop_var)
        counters = self._counters(loop, start)
        node = LoopN(name, extent, int(stages) if stages is not None else None)
        body: List[Any] = []
        self._stmt(loop.body, actor, body)
        for var in counters:
            self.env.loads.pop(var, None)
        self.loop_stack.pop()
        if any(a != actor for a, _item in body):
            raise Refusal("THREAD_PARTITION_UNSUPPORTED", "loop %s contains a role branch" % name)
        node.body = [item for _a, item in body]
        node._start = start  # type: ignore[attr-defined]
        out.append((actor, node))

    def _trips(self, loop: Any) -> Any:
        """Constant trips, or trips that vary with the CTA index only (evaluated per CTA later)."""

        tirx = self.tirx
        extent = self.env.subst(loop.extent)
        if isinstance(extent, tirx.IntImm):
            return int(extent.value)
        free: List[Any] = []
        tirx.stmt_functor.post_order_visit(tirx.Evaluate(extent), lambda n: free.append(n) if isinstance(n, tirx.Var) else None)
        if free and all(self._grid_entry(v) is not None for v in free):
            return CtaTrips(extent)
        raise Refusal("UNBOUND_SYMBOL", "loop extent of %s is not constant: %s (bind the symbols it uses)"
                      % (loop.loop_var.name, extent))

    def work_units(self, loops: List["LoopN"]) -> Tuple[Tuple[Any, ...], Tuple[Tuple[str, int], ...]]:
        """Group CTAs by the trip counts of CTA-dependent loops; runs follow the linear block id (x fastest)."""

        tirx = self.tirx
        order = {"blockIdx.x": 0, "blockIdx.y": 1, "blockIdx.z": 2}
        axes = sorted(self.grid, key=lambda e: order.get(self.grid_tags.get(e[0], ""), 9))
        total = _elements(tuple(e for _v, _n, e in axes))
        if total > 1 << 22:
            raise Refusal("WORK_GROUPS_UNSUPPORTED", "%d CTAs with CTA-dependent trip counts" % total)
        tables = []
        for loop in loops:
            used = [i for i, (var, _n, _e) in enumerate(axes) if self._refers([loop.trips.expr], var)]
            cache: Dict[Tuple[int, ...], int] = {}
            tables.append((loop.name, loop.trips.expr, used, cache))
        groups: "OrderedDict[Tuple[int, ...], str]" = OrderedDict()
        runs: List[List[Any]] = []
        index = [0] * len(axes)
        for _ in range(total):
            key = []
            for _name, expr, used, cache in tables:
                coords = tuple(index[i] for i in used)
                if coords not in cache:
                    value = self.env.analyzer.simplify(tirx.stmt_functor.substitute(
                        expr, {axes[i][0]: tirx.IntImm(axes[i][0].dtype, c) for i, c in zip(used, coords)}))
                    if not isinstance(value, tirx.IntImm):
                        raise Refusal("UNBOUND_SYMBOL", "trip count %s is not constant per CTA" % expr)
                    cache[coords] = max(0, int(value.value))
                key.append(cache[coords])
            gid = groups.setdefault(tuple(key), "g%d" % len(groups))
            if runs and runs[-1][0] == gid:
                runs[-1][1] += 1
            else:
                runs.append([gid, 1])
            for i in range(len(axes)):              # x fastest
                index[i] += 1
                if index[i] < axes[i][2]:
                    break
                index[i] = 0
        work_groups = tuple((gid, tuple((name, n) for (name, _e, _u, _c), n in zip(tables, key)))
                            for key, gid in groups.items())
        return work_groups, tuple((gid, count) for gid, count in runs)

    def _counters(self, loop: Any, start: int) -> List[Any]:
        """``x[0] = x[0] + 1`` in the body: ``x`` equals its initial value plus the iteration index."""

        tirx = self.tirx
        stmts = list(loop.body.seq) if isinstance(loop.body, tirx.SeqStmt) else [loop.body]
        found = []
        for stmt in stmts:
            if isinstance(stmt, tirx.BufferStore) and self._scope(stmt.buffer) == "local.var":
                value = stmt.value
                if isinstance(value, tirx.Add) and isinstance(value.a, tirx.BufferLoad) \
                        and value.a.buffer.same_as(stmt.buffer) and isinstance(value.b, tirx.IntImm) and int(value.b.value) == 1:
                    init = self.env.loads.get(stmt.buffer.data)
                    if init is None:
                        raise Refusal("COUNTER_UNSUPPORTED", "counter %s has no known initial value" % stmt.buffer.name)
                    self.env.loads[stmt.buffer.data] = init + (loop.loop_var - start)
                    found.append(stmt.buffer.data)
        return found

    def _slot_select(self, loop: Any, actor: Any, out: List[Any]) -> bool:
        """``for s in range(S): if stage == s: X`` executes X once with ``s := stage``."""

        tirx = self.tirx
        body = list(loop.body.seq) if isinstance(loop.body, tirx.SeqStmt) else [loop.body]
        var = loop.loop_var
        guarded = []
        for stmt in body:
            if not isinstance(stmt, tirx.IfThenElse) or stmt.else_case is not None or not isinstance(stmt.condition, tirx.EQ):
                return False
            cond = stmt.condition
            if isinstance(cond.b, tirx.Var) and cond.b.same_as(var):
                selector = cond.a
            elif isinstance(cond.a, tirx.Var) and cond.a.same_as(var):
                selector = cond.b
            else:
                return False
            guarded.append((selector, stmt.then_case))
        if not guarded:
            return False
        self.env.vmap[var] = guarded[0][0]
        for _selector, stmt in guarded:
            self._stmt(stmt, actor, out)
        del self.env.vmap[var]
        return True

    def _if(self, stmt: Any, actor: Any, out: List[Any]) -> None:
        if self.thread_var is not None and self._refers([self.env.subst(stmt.condition)], self.thread_var) \
                and self._element_body(stmt.then_case) \
                and (stmt.else_case is None or self._element_body(stmt.else_case)):
            self._predicated(stmt, actor, out)
            return
        interval = self._thread_interval(stmt.condition)
        if interval is not None:
            self.ws = True
            self._stmt(stmt.then_case, interval, out)
            if stmt.else_case is not None:
                lo, hi = interval
                rest = (hi, self.thread_extent) if lo == 0 else ((0, lo) if hi == self.thread_extent else None)
                if rest is None:
                    raise Refusal("THREAD_PARTITION_UNSUPPORTED", "else branch of a two-sided thread range")
                self._stmt(stmt.else_case, rest, out)
            return
        cond = self.env.subst(stmt.condition)
        if isinstance(cond, self.tirx.IntImm):
            branch = stmt.then_case if int(cond.value) else stmt.else_case
            if branch is not None:
                self._stmt(branch, actor, out)
            return
        raise Refusal("BRANCH_UNSUPPORTED", "condition %s is neither a thread range nor constant" % cond)

    def _predicated(self, stmt: Any, actor: Any, out: List[Any]) -> None:
        """Element statements under a thread predicate run on the selected threads; the selection
        must be an arithmetic progression (``tx == 0``, ``tx % 32 == 31``, ``tx < 64``)."""

        tirx = self.tirx
        lo, hi = actor if actor is not None else (0, self.thread_extent)
        base = self.thread_select
        candidates = range(lo, hi) if base is None else [base[0] + base[1] * k for k in range(base[2])]
        cond = self.env.subst(stmt.condition)
        chosen, rest = [], []
        for t in candidates:
            value = self.env.analyzer.simplify(tirx.stmt_functor.substitute(cond, {self.thread_var: tirx.IntImm(self.thread_var.dtype, t)}))
            if not isinstance(value, tirx.IntImm):
                raise Refusal("BRANCH_UNSUPPORTED", "condition %s depends on more than the thread index" % cond)
            (chosen if int(value.value) else rest).append(t)
        for selection, body in ((chosen, stmt.then_case), (rest, stmt.else_case)):
            if body is None or not selection:
                continue
            step = selection[1] - selection[0] if len(selection) > 1 else 1
            if any(t != selection[0] + k * step for k, t in enumerate(selection)):
                raise Refusal("BRANCH_UNSUPPORTED", "threads selected by %s are not evenly spaced" % cond)
            self.thread_select = (selection[0], step, len(selection))
            try:
                self._stmt(body, actor, out)
            finally:
                self.thread_select = base

    # -- operations ------------------------------------------------------------------

    def _evaluate(self, call: Any, actor: Any, out: List[Any]) -> None:
        tirx = self.tirx
        if not isinstance(call, tirx.Call):
            return
        name = _op_name(call)
        if actor is not None and name == "tl.named_barrier_arrive":
            out.append((actor, Op("barrier", self._unique("arrive_bar%d" % int(call.args[0])),
                                  barrier=("arrive", int(call.args[0])))))
            return
        if actor is not None and name == "tirx.tvm_storage_sync" and len(call.args) == 3:
            out.append((actor, Op("barrier", self._unique("sync_bar%d" % int(call.args[1])),
                                  barrier=("sync", int(call.args[1])))))
            return
        if name in _SYNC_CALLS:
            return
        if name in ("tl.tileop.copy", "tl.tileop.tma_copy", "tl.tileop.async_copy"):
            self._copy(call, name, actor, out)
        elif name in ("tl.tileop.gemm", "tl.tileop.wgmma_gemm", "tl.tileop.gemm_py"):
            self._gemm(call, actor, out)
        elif name == "tl.tileop.fill":
            buf, _ring = self._on_chip(self._region(call.args[0]), "fill", write=True)
            out.append((actor, Op("elementwise", self._unique("fill_" + buf.name), output=buf.name,
                                  ops=(("fill", 1.0, "cuda"),),
                                  compute_dtype=buf.dtype if buf.dtype.startswith(("fp", "bf")) else "fp32")))
        elif name == "tl.tileop.reduce":
            src, dst = self._region(call.args[0]), self._region(call.args[1])
            kind, dim, clear = str(call.args[2].value), int(call.args[3]), bool(call.args[4])
            a, _ra = self._on_chip(src, "reduce input")
            b, _rb = self._on_chip(dst, "reduce output", write=True)
            reducer_op = {"sum": "add", "max": "max", "min": "min", "abssum": "add", "absmax": "max"}.get(kind)
            if reducer_op is None:
                raise Refusal("REDUCE_UNSUPPORTED", "reducer %s" % kind)
            out.append((actor, Op("reduce", self._unique("reduce_%s_%s" % (kind, b.name)), inputs=(a.name,),
                                  output=b.name, axis=self._logical_axis(src, dim),
                                  reducer=kind if kind in ("sum", "max", "min") else "sum",
                                  ops=((reducer_op, 1.0, "cuda"),), accumulate=not clear)))
        else:
            raise Refusal("OPERATION_UNSUPPORTED", "no translation for call %s" % name)

    def _logical_axis(self, region: _Region, dim: int) -> int:
        shape = [self.env.const(d, "shape") for d in region.buffer.shape]
        kept = [d for d, extent in enumerate(shape) if extent != 1]
        if self.stage_kind.get(self._name(region.buffer)):
            kept = [d for d in kept if d != 0]
        if dim not in kept:
            raise Refusal("REDUCE_UNSUPPORTED", "reduction over a unit dimension")
        return kept.index(dim)

    def _on_chip(self, region: _Region, what: str, write: bool = False) -> Tuple[Buf, Optional[Tuple[str, int]]]:
        """Buffer of an on-chip operand and, for a ring slot inside a loop, (access, slot offset)."""

        tirx = self.tirx
        scope = self._scope(region.buffer)
        if not (scope.startswith("shared") or scope.startswith("local")):
            raise Refusal("OPERAND_UNSUPPORTED", "%s operand %s is in %s" % (what, region.buffer.name, scope))
        kind = self.stage_kind.get(self._name(region.buffer))
        if kind is None:
            return self._buf(region.buffer), None
        index = self.env.subst(region.mins[0])
        if kind == "split":
            if not isinstance(index, tirx.IntImm):
                raise Refusal("STAGE_VIEW_UNSUPPORTED", "%s stage index %s" % (region.buffer.name, index))
            return self._buf(region.buffer, int(index.value)), None
        buf = self._buf(region.buffer)
        if isinstance(index, tirx.IntImm):
            return buf, None                       # a peeled iteration outside the loop
        slots = buf.slots or 1
        if isinstance(index, tirx.FloorMod) and isinstance(index.b, tirx.IntImm) and int(index.b.value) == slots:
            terms = _affine(index.a, list(self.loop_stack))
            if terms is not None and len(terms[0]) == 1 and list(terms[0].values())[0] == 1 \
                    and list(terms[0])[0].same_as(self.loop_stack[-1]):
                return buf, ("write" if write else "read", terms[1] % slots)
        raise Refusal("RING_INDEX_UNSUPPORTED", "%s slot index %s is not (loop + c) %% %d" % (buf.name, index, slots))

    def _copy(self, call: Any, name: str, actor: Any, out: List[Any]) -> None:
        src, dst = self._region(call.args[0]), self._region(call.args[1])
        src_global, dst_global = self._scope(src.buffer) == "global", self._scope(dst.buffer) == "global"
        engine = "tma" if name == "tl.tileop.tma_copy" else ("cp_async" if name == "tl.tileop.async_copy" else None)
        if src_global and not dst_global:
            buf, ring = self._on_chip(dst, "copy destination", write=True)
            sl = self._slice(src, buf.shape, "copy %s -> %s" % (src.buffer.name, buf.name))
            out.append((actor, Op("load", self._unique("load_" + src.buffer.name), src=sl, dst=buf.name, engine=engine, ring=ring)))
        elif dst_global and not src_global:
            buf, ring = self._on_chip(src, "copy source")
            sl = self._slice(dst, buf.shape, "copy %s -> %s" % (buf.name, dst.buffer.name))
            out.append((actor, Op("store", self._unique("store_" + dst.buffer.name), src=buf.name, dst=sl, engine=engine, ring=ring)))
        elif not src_global and not dst_global:
            a, ring_a = self._on_chip(src, "copy source")
            b, ring_b = self._on_chip(dst, "copy destination", write=True)
            out.append((actor, Op("move", self._unique("copy_%s_%s" % (a.name, b.name)), src=a.name, dst=b.name,
                                  engine=engine, ring=ring_a or ring_b)))
        else:
            raise Refusal("OPERATION_UNSUPPORTED", "global-to-global copy")

    def _gemm(self, call: Any, actor: Any, out: List[Any]) -> None:
        tirx = self.tirx
        a, b, c = (self._region(call.args[i]) for i in range(3))
        if bool(call.args[3]):
            raise Refusal("GEMM_LAYOUT_UNSUPPORTED", "transposed A operand")
        clear = call.args[9]
        if isinstance(clear, (bool, int)):
            accumulate = not bool(clear)
        else:
            clear = self.env.subst(clear)
            # a constant clear overwrites; a condition such as ``k == 0`` clears only the first iteration
            accumulate = not (isinstance(clear, tirx.IntImm) and int(clear.value))
        (ba, ra), (bb, rb) = self._on_chip(a, "gemm A"), self._on_chip(b, "gemm B")
        bc, _rc = self._on_chip(c, "gemm C", write=True)
        out.append((actor, Op("gemm", self._unique("gemm_%s_%s" % (ba.name, bb.name)), inputs=(ba.name, bb.name),
                              output=bc.name, accumulate=accumulate, transpose_b=bool(call.args[4]),
                              ring=_merge_ring(ra, rb))))

    # -- element-wise statements -----------------------------------------------------
    #
    # An element statement ``t[idx] = f(...)`` runs once per point of its lane space:
    # the enclosing T.Parallel lanes (distributed over the CTA), element loops written
    # per thread (vectorized, unrolled or serial loops of stores) and, in code outside
    # T.Parallel that indexes by threadIdx.x or writes thread-private storage, the
    # threads themselves. Thread-private ``local`` arrays are per-thread arrays and a
    # ``local.var`` that carries data is one value per lane (or per thread).

    def _data_vars(self, func: Any) -> set:
        """``local.var`` buffers that hold data (float, or computed from a non-scalar buffer), not indices."""

        tirx = self.tirx
        found: set = set()

        def visit(node):
            if isinstance(node, tirx.BufferStore) and self._scope(node.buffer) == "local.var":
                reads: List[Any] = []
                tirx.stmt_functor.post_order_visit(
                    tirx.Evaluate(node.value),
                    lambda n: reads.append(n) if isinstance(n, tirx.BufferLoad) and self._scope(n.buffer) != "local.var" else None)
                if not str(node.buffer.dtype).startswith(("int", "uint", "bool")) or reads:
                    found.add(node.buffer.data)

        tirx.stmt_functor.post_order_visit(func.body, visit)
        return found

    def _store(self, stmt: Any, actor: Any, out: List[Any]) -> None:
        tirx = self.tirx
        if self._scope(stmt.buffer) == "local.var" and stmt.buffer.data not in self.data_vars:
            value = stmt.value
            increment = isinstance(value, tirx.Add) and isinstance(value.a, tirx.BufferLoad) \
                and value.a.buffer.same_as(stmt.buffer)
            if not increment:
                self.env.loads[stmt.buffer.data] = self.env.subst(value)
            return
        self._elementwise(stmt, stmt.value, [], actor, out)

    def _element_body(self, stmt: Any) -> bool:
        """Only element statements: stores, bindings, predicates, fences and loops over them."""

        tirx = self.tirx
        kinds = (tirx.ForKind.PARALLEL, tirx.ForKind.SERIAL, tirx.ForKind.VECTORIZED, tirx.ForKind.UNROLLED)
        if isinstance(stmt, tirx.For):
            return stmt.kind in kinds and self._element_body(stmt.body)
        if isinstance(stmt, tirx.SeqStmt):
            return all(self._element_body(x) for x in stmt.seq)
        if type(stmt).__name__ in ("LetStmt", "Bind"):
            return getattr(stmt, "body", None) is None or self._element_body(stmt.body)
        if isinstance(stmt, tirx.IfThenElse):
            return self._element_body(stmt.then_case) and (stmt.else_case is None or self._element_body(stmt.else_case))
        if isinstance(stmt, tirx.Evaluate):
            return isinstance(stmt.value, tirx.Call) and _op_name(stmt.value) in _SYNC_CALLS
        if isinstance(stmt, tirx.BufferStore):
            return not (self._scope(stmt.buffer) == "local.var" and stmt.buffer.data not in self.data_vars)
        return False

    def _element_nest(self, loop: Any) -> bool:
        """A loop nest that only distributes element statements over lanes: every serial loop in it
        has independent iterations, apart from accumulations ``t = t (+|max|min) f``."""

        tirx = self.tirx
        if not self._element_body(loop):
            return False
        serial: List[Any] = []
        tirx.stmt_functor.post_order_visit(loop, lambda n: serial.append(n) if isinstance(n, tirx.For) and n.kind in (
            tirx.ForKind.SERIAL, tirx.ForKind.UNROLLED) else None)
        return not any(self._carried(item) for item in serial)

    def _carried(self, loop: Any) -> bool:
        """Iterations of ``loop`` depend on each other (a scan or recurrence), other than by accumulation."""

        tirx = self.tirx
        binds: Dict[Any, Any] = {}
        stores: List[Any] = []

        def visit(node):
            if type(node).__name__ in ("LetStmt", "Bind"):
                binds[node.var] = node.value
            elif isinstance(node, tirx.BufferStore):
                stores.append(node)

        tirx.stmt_functor.post_order_visit(loop.body, visit)

        def expand(expr: Any) -> Any:
            for _ in range(8):
                new = tirx.stmt_functor.substitute(expr, binds) if binds else expr
                if new.same_as(expr):
                    break
                expr = new
            return expr

        accumulated = set()
        for store in stores:
            value = expand(store.value)
            indices = [expand(i) for i in store.indices]
            if self._refers(indices, loop.loop_var):
                if self._reads_other(store.buffer, indices, value):
                    return True
                continue
            try:
                self._combiner(store, value)
            except Refusal:
                return True
            accumulated.add(store.buffer.data)
        for store in stores:
            if store.buffer.data in accumulated:
                continue
            reads: List[Any] = []
            tirx.stmt_functor.post_order_visit(tirx.Evaluate(expand(store.value)), lambda n: reads.append(n) if isinstance(
                n, tirx.BufferLoad) and n.buffer.data in accumulated else None)
            if reads:
                return True                              # it observes partial sums: a scan
        return False

    def _reads_other(self, buffer: Any, indices: List[Any], value: Any) -> bool:
        tirx = self.tirx
        found: List[Any] = []

        def visit(node):
            if isinstance(node, tirx.BufferLoad) and node.buffer.same_as(buffer):
                if len(node.indices) != len(indices) or not all(
                        _tvm().ir.structural_equal(self.env.analyzer.simplify(a), self.env.analyzer.simplify(b))
                        for a, b in zip(node.indices, indices)):
                    found.append(node)

        tirx.stmt_functor.post_order_visit(tirx.Evaluate(value), visit)
        return bool(found)

    def _parallel(self, loop: Any, actor: Any, out: List[Any]) -> None:
        stores: List[Tuple[Any, Any, List[Tuple[Any, int, str]]]] = []
        self._flatten(loop, None, [], stores)
        for store, value, lanes in stores:
            self._elementwise(store, value, lanes, actor, out)

    def _flatten(self, stmt: Any, cond: Any, lanes: List[Tuple[Any, int, str]],
                 stores: List[Tuple[Any, Any, List[Tuple[Any, int, str]]]]) -> None:
        """Element stores with their lanes; ``if`` becomes a predicated store (both paths issue in SIMT)."""

        tirx = self.tirx
        if isinstance(stmt, tirx.For):
            extent = self.env.const(stmt.extent, "element loop extent of %s" % stmt.loop_var.name)
            self.env.analyzer.bind(stmt.loop_var, _tvm().ir.Range.from_min_extent(0, max(extent, 1)))
            if self.env.const(stmt.min, "element loop start") != 0:
                self.env.vmap[stmt.loop_var] = stmt.loop_var + stmt.min
            kind = "parallel" if stmt.kind == tirx.ForKind.PARALLEL else "serial"
            self._flatten(stmt.body, cond, lanes + [(stmt.loop_var, extent, kind)], stores)
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                self._flatten(child, cond, lanes, stores)
        elif type(stmt).__name__ in ("LetStmt", "Bind"):
            self.env.vmap[stmt.var] = stmt.value
            if getattr(stmt, "body", None) is not None:
                self._flatten(stmt.body, cond, lanes, stores)
        elif isinstance(stmt, tirx.IfThenElse):
            c = stmt.condition if cond is None else tirx.And(cond, stmt.condition)
            self._flatten(stmt.then_case, c, lanes, stores)
            if stmt.else_case is not None:
                n = tirx.Not(stmt.condition)
                self._flatten(stmt.else_case, n if cond is None else tirx.And(cond, n), lanes, stores)
        elif isinstance(stmt, tirx.BufferStore):
            value = stmt.value
            if cond is not None:
                value = tirx.Select(cond, value, tirx.BufferLoad(stmt.buffer, stmt.indices))
            stores.append((stmt, value, list(lanes)))
        elif isinstance(stmt, tirx.Evaluate):
            return                                              # a fence inside an element loop
        else:
            raise Refusal("PARALLEL_UNSUPPORTED", "element loop body contains %s" % type(stmt).__name__)

    def _thread_lane(self, actor: Any) -> Tuple[Any, int, Any]:
        """A lane over the threads that run a statement and the expression threadIdx.x takes on it."""

        lo, hi = actor if actor is not None else (0, self.thread_extent or 1)
        var = self.tirx.Var("tx_lane", self.thread_var.dtype if self.thread_var is not None else "int32")
        if self.thread_select is not None:
            first, step, count = self.thread_select
            return var, count, var * step + first
        return var, hi - lo, var + lo

    def _threads(self, actor: Any) -> int:
        lo, hi = actor if actor is not None else (0, self.thread_extent or 1)
        return hi - lo

    def _refers(self, exprs: Sequence[Any], var: Any) -> bool:
        found = []
        for expr in exprs:
            self.tirx.stmt_functor.post_order_visit(
                self.tirx.Evaluate(expr), lambda n: found.append(n) if isinstance(n, self.tirx.Var) and n.same_as(var) else None)
        return bool(found)

    def _elementwise(self, store: Any, value: Any, lanes: List[Tuple[Any, int, str]], actor: Any,
                     out: List[Any]) -> None:
        """One element statement -> loads of global reads, element work (and a reduce for an
        accumulation ``t = t + f`` / ``max`` / ``min`` over lanes ``t`` is not indexed by), and a
        store for a global target."""

        tirx = self.tirx
        indices = [self.env.subst(i) for i in store.indices]
        value = self.env.subst(value)
        parallel = [l for l in lanes if l[2] == "parallel"]
        scope = self._scope(store.buffer)
        is_data_var = scope == "local.var"
        if not parallel and self.thread_var is not None:
            per_thread = scope == "local" or is_data_var or self._refers(indices + [value], self.thread_var)
            if per_thread:
                var, extent, expr = self._thread_lane(actor)
                sub = {self.thread_var: expr}
                indices = [self.env.analyzer.simplify(tirx.stmt_functor.substitute(i, sub)) for i in indices]
                value = tirx.stmt_functor.substitute(value, sub)
                self.env.analyzer.bind(var, _tvm().ir.Range.from_min_extent(0, extent))
                lanes = [(var, extent, "thread")] + lanes
        active = [l for l in lanes if l[1] > 1]
        space = tuple(l[1] for l in active) or (1,)

        # the target and the lanes that index it
        target_kind, target, target_lanes = self._element_target(store, indices, lanes, actor)
        reduced = [l for l in active if not any(l[0].same_as(v[0]) for v in target_lanes)]
        combine = None
        if reduced:
            combine = self._combiner(store, value)
            value = combine[1]
        elif self._reads_other_element(store, indices, value, lanes):
            raise Refusal("PARALLEL_UNSUPPORTED", "%s reads another element of itself across lanes "
                          "(a recurrence)" % store.buffer.name)
        loads: "OrderedDict[str, Any]" = OrderedDict()

        def collect(node):
            if isinstance(node, tirx.BufferLoad) and not (
                    self._scope(node.buffer) == "local.var" and node.buffer.data not in self.data_vars):
                loads.setdefault("%s%s" % (self._name(node.buffer), [str(i) for i in node.indices]), node)
        tirx.stmt_functor.post_order_visit(tirx.Evaluate(value), collect)
        ops, compute = _count_ops(value, self.env)
        if not ops:
            ops = (("mov", 1.0, "cuda"),)
        out_shape = target[1].shape if target_kind == "global" else target.shape
        reads: List[str] = []
        rows: List[str] = []
        for load in loads.values():
            name, shape, by_first = self._element_operand(load, lanes, actor, out)
            if name in reads:
                continue
            reads.append(name)
            if by_first and not reduced and len(out_shape) >= 2 and shape[0] == out_shape[0]:
                rows.append(name)

        def mapped_for(shape: Tuple[int, ...]) -> Tuple[str, ...]:
            return tuple(r for r in reads if r not in rows and not _broadcasts(self.bufs[r].shape, shape))

        if reduced:
            if target_kind == "global":
                raise Refusal("PARALLEL_UNSUPPORTED", "accumulation into global %s" % store.buffer.name)
            red = _elements(tuple(l[1] for l in reduced))
            kept = _elements(tuple(l[1] for l in active if l not in reduced) or (1,))
            if kept != _elements(target.shape):
                raise Refusal("PARALLEL_UNSUPPORTED", "%d lanes accumulate into %s%r" % (kept, target.name, target.shape))
            partial = self._hidden(target.name + "__terms", (red,) + target.shape, "float32" if compute == "fp32" else store.buffer.dtype)
            out.append((actor, Op("elementwise", self._unique("ew_" + partial.name), inputs=tuple(reads), output=partial.name,
                                  ops=ops, compute_dtype=compute, mapped_inputs=mapped_for(partial.shape))))
            reducer = combine[0]
            out.append((actor, Op("reduce", self._unique("reduce_%s_%s" % (reducer, target.name)), inputs=(partial.name,),
                                  output=target.name, axis=0, reducer=reducer,
                                  ops=(({"sum": "add"}.get(reducer, reducer), 1.0, "cuda"),), accumulate=True)))
            return
        if target_kind == "global":
            sl, hidden = target
            out.append((actor, Op("elementwise", self._unique("ew_" + hidden.name), inputs=tuple(reads), output=hidden.name,
                                  ops=ops, compute_dtype=compute, row_inputs=tuple(rows),
                                  mapped_inputs=mapped_for(hidden.shape))))
            out.append((actor, Op("store", self._unique("store_" + store.buffer.name), src=hidden.name, dst=sl, engine="ldst")))
            return
        elements, written = _elements(target.shape), _elements(space)
        if written > elements:
            raise Refusal("PARALLEL_UNSUPPORTED", "%d lanes write %s%r" % (written, target.name, target.shape))
        if written < elements:      # a partial update: the work of the written elements, spread over the buffer
            ops = tuple((cls, count * written / elements, engine) for cls, count, engine in ops)
        out.append((actor, Op("elementwise", self._unique("ew_" + target.name), inputs=tuple(reads), output=target.name,
                              ops=ops, compute_dtype=compute, row_inputs=tuple(rows),
                              mapped_inputs=mapped_for(target.shape))))

    def _element_target(self, store: Any, indices: List[Any], lanes: List[Tuple[Any, int, str]], actor: Any):
        """(kind, buffer or (slice, register tile), lanes that index the target)."""

        scope = self._scope(store.buffer)
        if scope == "global":
            sl, shape, used = self._pointwise_slice(store.buffer, indices, lanes)
            return "global", (sl, self._hidden("%s__reg" % store.buffer.name, shape, store.buffer.dtype)), used
        return "local", self._lane_buffer(store.buffer, lanes, actor), self._index_lanes(store.buffer, indices, lanes)

    def _lane_buffer(self, buffer: Any, lanes: List[Tuple[Any, int, str]], actor: Any = None) -> Buf:
        """Logical buffer of an element access: thread-private arrays gain a thread dimension, a data
        ``local.var`` is one value per lane of its T.Parallel nest (or per thread)."""

        scope = self._scope(buffer)
        if scope == "local.var":
            parallel = tuple(l[1] for l in lanes if l[2] == "parallel" and l[1] > 1)
            threads = (self._threads(actor),) if any(l[2] == "thread" for l in lanes) else ()
            return self._hidden(self._name(buffer) + "__lanes", parallel or threads or (1,), buffer.dtype)
        buf = self._buf(buffer)
        if scope == "local":
            if not any(l[2] == "thread" for l in lanes):
                raise Refusal("PARALLEL_UNSUPPORTED", "thread-private %s used outside per-thread code" % buffer.name)
            name = buf.name + "__threads"
            shape = (self._threads(actor),) + tuple(e for e in buf.shape if e != 1)
            if name not in self.bufs:
                self.bufs[name] = Buf(name, "register", shape, buf.dtype, base=name)
            elif self.bufs[name].shape != shape:
                raise Refusal("BUFFER_VIEW_CONFLICT", "%s is used by %d and %d threads" % (buffer.name, self.bufs[name].shape[0], shape[0]))
            return self.bufs[name]
        return buf

    def _index_lanes(self, buffer: Any, indices: List[Any], lanes: List[Tuple[Any, int, str]]) -> List[Tuple[Any, int, str]]:
        scope = self._scope(buffer)
        if scope == "local.var":
            parallel = [l for l in lanes if l[2] == "parallel"]
            return parallel or [l for l in lanes if l[2] == "thread"]
        used = [l for l in lanes if self._refers(indices, l[0])]
        if scope == "local":
            used = [l for l in lanes if l[2] == "thread"] + [l for l in used if l[2] != "thread"]
        return used

    def _element_operand(self, load: Any, lanes: List[Tuple[Any, int, str]], actor: Any,
                         out: List[Any]) -> Tuple[str, Tuple[int, ...], bool]:
        """Register or on-chip tile an element statement reads; a global read becomes a load."""

        indices = [self.env.subst(i) for i in load.indices]
        scope = self._scope(load.buffer)
        if scope == "global":
            sl, shape, _used = self._pointwise_slice(load.buffer, indices, lanes)
            hidden = self._hidden("%s__reg" % load.buffer.name, shape, load.buffer.dtype)
            out.append((actor, Op("load", self._unique("load_" + load.buffer.name), src=sl, dst=hidden.name, engine="ldst")))
            return hidden.name, hidden.shape, False
        buf = self._lane_buffer(load.buffer, lanes, actor)
        active = [l for l in lanes if l[1] > 1]
        by_first = scope not in ("local", "local.var") and len(indices) == 1 and bool(active) \
            and isinstance(indices[0], self.tirx.Var) and indices[0].same_as(active[0][0])
        return buf.name, buf.shape, by_first

    def _reads_other_element(self, store: Any, indices: List[Any], value: Any,
                             lanes: List[Tuple[Any, int, str]]) -> bool:
        """The value reads an element of the target that another lane writes (a recurrence across lanes).
        Another element in a dimension that no lane indexes belongs to another loop iteration: carried."""

        tirx = self.tirx
        found = []
        lane_vars = [l[0] for l in lanes if l[1] > 1]

        def visit(node):
            if isinstance(node, tirx.BufferLoad) and node.buffer.same_as(store.buffer) and len(node.indices) == len(indices):
                for a, b in zip(node.indices, indices):
                    a, b = self.env.analyzer.simplify(self.env.subst(a)), self.env.analyzer.simplify(b)
                    if not _tvm().ir.structural_equal(a, b) and any(self._refers([b], v) for v in lane_vars):
                        found.append(node)

        tirx.stmt_functor.post_order_visit(tirx.Evaluate(value), visit)
        if self._scope(store.buffer) == "shared.dyn" or self._scope(store.buffer).startswith("shared"):
            carried = []
            tirx.stmt_functor.post_order_visit(tirx.Evaluate(value), lambda n: carried.append(n) if isinstance(
                n, tirx.BufferLoad) and n.buffer.same_as(store.buffer) else None)
            if carried and self.loop_stack:
                note = ("%s carries values between iterations of loop %s through shared memory; TileSight orders "
                        "operations within an iteration, so that chain's latency is not priced"
                        % (store.buffer.name, self.loop_vars.get(self.loop_stack[-1], "?")))
                if note not in self.notes:
                    self.notes.append(note)
        return bool(found) and self._scope(store.buffer) != "local.var"

    def _combiner(self, store: Any, value: Any) -> Tuple[str, Any]:
        """``t = t + f`` / ``max(t, f)`` / ``min(t, f)`` -> (reducer, f)."""

        tirx = self.tirx

        def is_target(node: Any) -> bool:
            return isinstance(node, tirx.BufferLoad) and node.buffer.same_as(store.buffer)

        def mentions_target(expr: Any) -> bool:
            hits: List[Any] = []
            tirx.stmt_functor.post_order_visit(tirx.Evaluate(expr), lambda n: hits.append(n) if is_target(n) else None)
            return bool(hits)

        select = value if isinstance(value, tirx.Select) else (
            tirx.Select(*value.args) if isinstance(value, tirx.Call) and _op_name(value) == "tirx.if_then_else" else None)
        if select is not None and is_target(select.false_value):
            if not mentions_target(select.true_value) and mentions_target(select.condition):
                # t = c(t, f) ? f : t keeps a running extremum (max, min, argmax with NaN rules)
                return "max", tirx.Select(select.condition, select.true_value, select.true_value)
            if not mentions_target(select.condition):                     # if c: t = t (+) f
                reducer, term = self._combiner(store, select.true_value)
                return reducer, tirx.Select(select.condition, term, tirx.const(0, term.dtype))
        for kind, reducer in ((tirx.Add, "sum"), (tirx.Max, "max"), (tirx.Min, "min")):
            if isinstance(value, kind):
                for mine, other in ((value.a, value.b), (value.b, value.a)):
                    if is_target(mine):
                        return reducer, other
        if isinstance(value, tirx.Call) and _op_name(value) in ("tl.max_nan", "tl.min_nan") and len(value.args) == 2:
            for mine, other in ((value.args[0], value.args[1]), (value.args[1], value.args[0])):
                if is_target(mine):
                    return ("max" if "max" in _op_name(value) else "min"), other
        raise Refusal("PARALLEL_UNSUPPORTED", "%s is written by fewer lanes than run and is not an "
                      "accumulation (t = t + f, max or min)" % store.buffer.name)

    def _hidden(self, base: str, shape: Tuple[int, ...], dtype: Any) -> Buf:
        """A register tile the IR keeps implicit (element-wise global access, accumulation terms, lane scalars)."""

        shape = tuple(e for e in shape if e != 1) or (1,)
        name, n = base, 0
        while name in self.bufs and (self.bufs[name].shape != shape or self.bufs[name].dtype != _dtype(dtype)):
            n += 1
            name = "%s%d" % (base, n)
        self.bufs.setdefault(name, Buf(name, "register", shape, _dtype(dtype), base=name))
        return self.bufs[name]

    def _pointwise_slice(self, buffer: Any, indices: Sequence[Any], lanes: List[Tuple[Any, int, str]]
                         ) -> Tuple[Slice, Tuple[int, ...], List[Tuple[Any, int, str]]]:
        """Global element access -> tensor slice, the tile shape it covers and the lanes that index it.

        Per dimension the index is ``axis*n + lanes`` (a tile), ``lanes`` (the whole
        dimension; lanes beyond it are masked), ``axis`` (a pick), or a constant
        range. Several lanes in one dimension must form one contiguous range
        (``i*8 + j`` with ``j < 8``). An index that is affine only as a flat
        row-major offset (``x[e // c, e % c]``) reads the tensor as one dimension.
        """

        tirx = self.tirx
        tensor = self._declared_tensor(buffer)
        unit = {l[0]: tirx.IntImm(l[0].dtype, 0) for l in lanes if l[1] == 1}
        indices = [self._split_axes(self.env.subst(tirx.stmt_functor.substitute(i, unit) if unit else i)) for i in indices]
        shape = [self.env.const(d, "shape") for d in buffer.shape]
        try:
            return self._element_slice(tensor, shape, indices, lanes)
        except Refusal as refusal:
            if refusal.code != "INDEX_UNSUPPORTED" or len(indices) < 2:
                raise
            strides = [1] * len(shape)
            for d in range(len(shape) - 2, -1, -1):
                strides[d] = strides[d + 1] * shape[d + 1]
            declared = [self.env.const(x, "stride") for x in buffer.strides] if len(buffer.strides) else strides
            if declared != strides:
                raise
            flat = self.env.analyzer.simplify(sum((i * k for i, k in zip(indices, strides)), tirx.IntImm(indices[0].dtype, 0)))
            try:
                return self._element_slice(self._flat(tensor, "%s element access" % tensor.name), [_elements(shape)], [flat], lanes)
            except Refusal:
                raise refusal

    def _split_lanes(self, indices: List[Any], lanes: List[Tuple[Any, int, str]]
                     ) -> Tuple[List[Any], List[Tuple[Any, int, str]]]:
        """``i // c`` and ``i % c`` of a lane ``i`` (c dividing its extent) become two lanes of extents
        n/c and c; several lanes then read one element (``i // 2``) or one lane walks a row (``tx // 32``)."""

        tirx = self.tirx
        derived: Dict[Tuple[int, int, bool], Tuple[Any, int, str]] = {}
        extra: List[Tuple[Any, int, str]] = []

        def post(node):
            if isinstance(node, (tirx.FloorDiv, tirx.FloorMod)) and isinstance(node.b, tirx.IntImm) and isinstance(node.a, tirx.Var):
                lane = next((l for l in lanes if l[0].same_as(node.a) and l[1] > 1), None)
                c = int(node.b.value)
                if lane is None or c <= 1 or lane[1] % c:
                    return None
                is_div = isinstance(node, tirx.FloorDiv)
                key = (id(lane[0]), c, is_div)
                if key not in derived:
                    var = tirx.Var("%s_%s%d" % (lane[0].name, "div" if is_div else "mod", c), lane[0].dtype)
                    derived[key] = (var, lane[1] // c if is_div else c, lane[2])
                    extra.append(derived[key])
                    self.lane_origin[var] = lane
                return derived[key][0]
            return None

        out = [tirx.stmt_functor.ir_transform(tirx.Evaluate(i), None, post, ["tirx.FloorDiv", "tirx.FloorMod"]).value
               for i in indices]
        return out, extra

    def _element_slice(self, tensor: Tensor, shape: List[int], indices: List[Any], lanes: List[Tuple[Any, int, str]]
                       ) -> Tuple[Slice, Tuple[int, ...], List[Tuple[Any, int, str]]]:
        indices, extra = self._split_lanes(list(indices), lanes)
        active = [l for l in lanes if l[1] > 1] + extra
        lane_vars = [l[0] for l in active]
        items: List[Any] = []
        tile: List[int] = []
        used: List[Tuple[Any, int, str]] = []
        box: List[Optional[Tuple[str, int, int]]] = [None] * len(indices)
        what = "%s element access" % tensor.name
        for dim, index in enumerate(indices):
            if dim in tensor.squeeze:
                continue
            terms = _affine(index, self._axis_vars() + lane_vars)
            if terms is None:
                raise Refusal("INDEX_UNSUPPORTED", "%s index %s is not affine" % (tensor.name, index))
            full = shape[dim]
            lane_terms = sorted(((next(l for l in active if l[0].same_as(v)), c) for v, c in terms[0].items()
                                 if any(v.same_as(x) for x in lane_vars)), key=lambda e: e[1])
            axis = {v: c for v, c in terms[0].items() if not any(v.same_as(x) for x in lane_vars)}
            offset = terms[1]
            if not lane_terms:
                if not axis:
                    if not 0 <= offset < full:
                        raise Refusal("INDEX_UNSUPPORTED", "%s[dim %d] is the constant %s" % (tensor.name, dim, offset))
                    box[dim] = ("range", offset, 1)          # the view drops this dimension...
                    items.append(("const", dim))             # ...unless nothing else spans the register tile
                    continue
                axes, first, count = self._pick_range(index, what)
                if (first, count) != (0, full):
                    if first < 0 or first + count > full:
                        raise Refusal("INDEX_UNSUPPORTED", "%s[dim %d] picks %d of %d at %d" % (tensor.name, dim, count, full, first))
                    box[dim] = ("range", first, count)
                items.append(("pick", axes))
                continue
            stride = 1
            for lane, coeff in lane_terms:
                if coeff != stride:
                    raise Refusal("INDEX_UNSUPPORTED", "%s index %s is not one contiguous lane range" % (tensor.name, index))
                stride *= lane[1]
            for lane, _c in lane_terms:
                origin = next((o for v, o in self.lane_origin.items() if v.same_as(lane[0])), lane)
                if not any(origin[0].same_as(u[0]) for u in used):
                    used.append(origin)
            if not axis:
                if offset == 0 and stride >= full:
                    if stride > full:
                        note = "%s: %d lanes cover the %d elements of dim %d (the rest are masked)" % (what, stride, full, dim)
                        if note not in self.notes:
                            self.notes.append(note)
                    items.append(("full",))
                    tile.append(full)
                elif 0 <= offset and offset + stride <= full:
                    box[dim] = ("range", offset, stride)
                    items.append(("full",))
                    tile.append(stride)
                else:
                    raise Refusal("INDEX_UNSUPPORTED", "%s[dim %d] covers %d of %d at %d" % (tensor.name, dim, stride, full, offset))
                continue
            if len(axis) != 1:
                raise Refusal("INDEX_UNSUPPORTED", "%s index %s is not axis*tile + lanes" % (tensor.name, index))
            (var, coeff), = axis.items()
            if coeff == stride:
                box[dim] = _tile_cover(offset, stride, self._axis_extent(var), full)
                items.append(("tile", self._axis_name(var)))
            elif coeff % stride == 0 and offset % stride == 0 and 0 <= offset < coeff:
                box[dim] = ("blocks", coeff // stride, offset // stride)
                items.append(("tile", self._axis_name(var)))
            else:
                raise Refusal("INDEX_UNSUPPORTED", "%s index %s is not axis*tile + lanes" % (tensor.name, index))
            tile.append(stride)
        if not any(box):
            if not tensor.name.endswith("@flat"):
                self._whole(tensor, what)
            return Slice(tensor.name, tuple(i for i in items if i[0] != "const")), tuple(tile) or (1,), used
        if tensor.name.endswith("@flat"):
            raise Refusal("INDEX_UNSUPPORTED", "%s: a part of the flat view" % what)
        consts = [i for i, item in enumerate(items) if item[0] == "const"]
        if consts and all(item[0] in ("pick", "const") for item in items):
            keep = consts[-1]                            # one element: keep a dimension of extent 1
            box[items[keep][1]] = ("cell", box[items[keep][1]][1], 1)
            items[keep] = ("full",)
            tile = [1] + tile
        items = [item for item in items if item[0] != "const"]
        return Slice(self._view(tensor, box, what), tuple(items)), tuple(tile) or (1,), used


def _tile_cover(offset: int, tile: int, count: Optional[int], full: int) -> Optional[Tuple[str, int, int]]:
    """None for tiles of the whole dimension, a range view for tiles of part of it; raises if neither."""

    if count is None or (offset == 0 and full <= count * tile < full + tile):
        return None
    if offset % tile == 0 and 0 <= offset and offset + count * tile <= full:
        return ("range", offset, count * tile)
    raise Refusal("INDEX_UNSUPPORTED", "%d tiles of %d at %d do not fit a dimension of %d" % (count, tile, offset, full))


def _disjoint(a: Optional[Tuple[str, int, int]], b: Optional[Tuple[str, int, int]], full: int) -> bool:
    """Two view specs of one dimension select disjoint elements."""

    a = None if a is None else (("range",) + tuple(a[1:]) if a[0] == "cell" else a)
    b = None if b is None else (("range",) + tuple(b[1:]) if b[0] == "cell" else b)
    if a is None or b is None or a[0] != b[0]:
        return False
    if a[0] == "range":
        return a[1] + a[2] <= b[1] or b[1] + b[2] <= a[1]
    return a[1] == b[1] and a[2] != b[2]


def _broadcasts(shape: Tuple[int, ...], target: Tuple[int, ...]) -> bool:
    if tuple(shape) == tuple(target) or _elements(shape) == 1:
        return True
    if len(shape) > len(target):
        return False
    return all(a == b or a == 1 for a, b in zip(reversed(shape), reversed(target)))


def _merge_ring(*rings: Optional[Tuple[str, int]]) -> Optional[Tuple[str, int]]:
    present = [r for r in rings if r is not None]
    if not present:
        return None
    offsets = {r[1] for r in present}
    if len(offsets) > 1:
        raise Refusal("RING_INDEX_UNSUPPORTED", "one operation reads ring slots of different iterations")
    return present[0]


# --------------------------------------------------------------------------- iteration offsets

def _ring_offset(ring: Tuple[str, int], slots: int) -> int:
    """Iteration of a ring access: a read sees the latest earlier write, a write fills a slot ahead."""

    access, c = ring
    r = c % slots
    if access == "write":
        return r
    return r - slots if r else 0


def _reads(op: Op) -> List[str]:
    names = list(op.inputs)
    if isinstance(op.src, str):
        names.append(op.src)
    if op.accumulate and op.output:
        names.append(op.output)
    return names


def _writes(op: Op) -> List[str]:
    if op.kind in ("gemm", "elementwise", "reduce"):
        return [op.output] if op.output else []
    return [op.dst] if isinstance(op.dst, str) else []


def _offsets(body: List[Op], bufs: Dict[str, Buf]) -> List[int]:
    """Iteration offset of every body operation relative to the loop variable."""

    offsets: List[Optional[int]] = []
    last_writer: Dict[str, int] = {}
    for position, op in enumerate(body):
        known: List[int] = []
        if op.ring is not None:
            ring_buffer = next((n for n in _reads(op) + _writes(op) if bufs[n].slots), None)
            known.append(_ring_offset(op.ring, bufs[ring_buffer].slots if ring_buffer else 1))
        for name in _reads(op):
            writer = last_writer.get(name)
            if writer is not None and offsets[writer] is not None:
                known.append(offsets[writer])
        offsets.append(max(known) if known else None)
        for name in _writes(op):
            last_writer[name] = position
    for position, op in enumerate(body):
        if offsets[position] is not None:
            continue
        outputs = set(_writes(op))
        consumer = next((offsets[j] for j in range(position + 1, len(body))
                         if outputs & set(_reads(body[j])) and offsets[j] is not None), None)
        offsets[position] = consumer if consumer is not None else 0
    return [int(o) for o in offsets]


def _by_name(ops: List[Op], names: List[str]) -> List[Op]:
    table = {op.name: op for op in ops}
    return [table[name] for name in names]


def _signature(op: Op, bufs: Dict[str, Buf]) -> Tuple[Any, ...]:
    return (op.kind, tuple(bufs[n].base or n for n in op.inputs), bufs[op.output].base if op.output else None)


def _reroll(items: List[Any], role: str, bufs: Dict[str, Buf], notes: List[str]) -> List[Any]:
    """Re-roll a hand-pipelined role loop (peeled first iteration, drained last) into uniform iterations."""

    loops = [i for i, x in enumerate(items) if isinstance(x, LoopN)]
    if len(loops) != 1:
        return [x for x in items if not (isinstance(x, Op) and x.kind == "barrier")]
    position = loops[0]
    loop: LoopN = items[position]
    if any(isinstance(x, LoopN) for x in loop.body):
        return items
    issued = list(loop.body)
    real = [op for op in issued if op.kind != "barrier"]
    offsets = _offsets(real, bufs)
    low, high = min(offsets), max(offsets)
    for op, offset in zip(real, offsets):
        op.window = offset              # role-local; normalized once roles are merged
    pre_counts: Dict[Tuple[str, int], int] = {}
    for x in items[:position]:
        if isinstance(x, Op) and x.kind == "barrier":
            pre_counts[x.barrier] = pre_counts.get(x.barrier, 0) + 1
    loop.meta[role] = {"start": getattr(loop, "_start", 0), "pre": pre_counts}
    # Markers: an arrive follows the operation issued before it; a sync gates the next one.
    statements = [op for _o, _i, op in sorted(((-o, i, op) for i, (op, o) in enumerate(zip(real, offsets))),
                                              key=lambda e: (e[0], e[1]))]
    for i, op in enumerate(issued):
        if op.kind != "barrier":
            continue
        if op.barrier[0] == "arrive":
            before = next((x for x in reversed(issued[:i]) if x.kind != "barrier"), None)
            if before is None:
                raise Refusal("BARRIER_UNSUPPORTED", "%s: arrive before any operation of the iteration" % role)
            op.window = before.window
            statements.insert(statements.index(before) + 1, op)
        else:
            gated = next((x for x in issued[i + 1:] if x.kind != "barrier"), None)
            if gated is None:
                raise Refusal("BARRIER_UNSUPPORTED", "%s: sync after every operation of the iteration" % role)
            op.output = gated.name   # the operation this wait releases
    loop.issue_order[role] = [op.name for op in issued if not (op.kind == "barrier" and op.barrier[0] == "sync")]
    loop.body = [op for op in statements]
    loop._syncs = [op for op in issued if op.kind == "barrier" and op.barrier[0] == "sync"]  # type: ignore[attr-defined]
    items = [x for x in items[:position] if not (isinstance(x, Op) and x.kind == "barrier")] + [loop] + \
        [x for x in items[position + 1:] if not (isinstance(x, Op) and x.kind == "barrier")]
    position = items.index(loop)
    if high == low:
        return items
    if high - low != 1:
        raise Refusal("PIPELINE_DEPTH_UNSUPPORTED", "%s: operations span %d iterations" % (role, high - low + 1))
    ahead = [_signature(op, bufs) for op in real if op.window == high and op.kind == "gemm"]
    behind = [_signature(op, bufs) for op in real if op.window == low and op.kind == "gemm"]
    pre, post = items[:position], items[position + 1:]
    pre_gemms = [i for i, x in enumerate(pre) if isinstance(x, Op) and x.kind == "gemm"]
    if not ahead or not pre_gemms:
        raise Refusal("PIPELINE_PEEL_UNSUPPORTED", "%s: no peeled first iteration to match" % role)
    peel = pre[pre_gemms[0]:]
    if [_signature(x, bufs) for x in peel if isinstance(x, Op) and x.kind == "gemm"] != ahead \
            or any(isinstance(x, LoopN) or x.kind in ("load", "store") for x in peel):
        raise Refusal("PIPELINE_PEEL_UNSUPPORTED", "%s: the operations before the loop are not its first iteration" % role)
    post_gemms = [i for i, x in enumerate(post) if isinstance(x, Op) and x.kind == "gemm"]
    if not behind or not post_gemms:
        raise Refusal("PIPELINE_PEEL_UNSUPPORTED", "%s: no drained last iteration to match" % role)
    drain = post[:post_gemms[len(behind) - 1] + 1]
    if [_signature(x, bufs) for x in drain if isinstance(x, Op) and x.kind == "gemm"] != behind \
            or any(isinstance(x, LoopN) or x.kind in ("load", "store") for x in drain):
        raise Refusal("PIPELINE_PEEL_UNSUPPORTED", "%s: the operations after the loop are not its last iteration" % role)
    folded = len(peel) + len(drain) - sum(1 for o in offsets if o in (high, low))
    notes.append("%s: first iteration peeled and last drained in the source; translated as %s uniform iterations "
                 "(%d peeled operations folded%s)" % (role, str(loop.trips + 1), len(peel) + len(drain),
                                                   "" if not folded else ", %+d against the loop body" % folded))
    loop.trips += 1
    return pre[:pre_gemms[0]] + [loop] + post[len(drain):]


# --------------------------------------------------------------------------- structuring

def _actor_names(walked: List[Tuple[Optional[Tuple[int, int]], Any]]) -> Dict[Tuple[int, int], str]:
    intervals: List[Tuple[int, int]] = []
    for actor, _item in walked:
        if actor is not None and actor not in intervals:
            intervals.append(actor)
    names: Dict[Tuple[int, int], str] = {}
    consumers = 0
    for interval in sorted(intervals):
        flat: List[Op] = []
        for actor, item in walked:
            if actor == interval:
                flat.extend(item.body if isinstance(item, LoopN) else [item])
        if flat and all(op.kind == "load" for op in flat if isinstance(op, Op)):
            names[interval] = "producer" if "producer" not in names.values() else "producer%d" % len(names)
        else:
            names[interval] = "consumer" if consumers == 0 else "consumer%d" % consumers
            consumers += 1
    return names


def _tag(item: Any, actor: str) -> Any:
    if isinstance(item, Op):
        item.actor = actor
        return item
    for attr in ("body", "prologue", "epilogue"):
        for x in getattr(item, attr):
            _tag(x, actor)
    return item


def _rename_axis(items: List[Any], old: str, new: str) -> None:
    for op in items:
        if isinstance(op, LoopN):
            _rename_axis(op.body + op.prologue + op.epilogue, old, new)
            continue
        for attr in ("src", "dst"):
            value = getattr(op, attr)
            if isinstance(value, Slice):
                setattr(op, attr, Slice(value.tensor, tuple(
                    ("tile", new) if item == ("tile", old) else
                    (("pick", tuple(new if a == old else a for a in item[1])) if item[0] == "pick" else item)
                    for item in value.items)))


def _structure(walked: List[Tuple[Optional[Tuple[int, int]], Any]], walker: _Walker) -> Tuple[List[Any], Tuple[str, ...]]:
    actors = {actor for actor, _ in walked if actor is not None}
    if not actors:
        items = [item for _a, item in walked]
        for item in items:
            if isinstance(item, LoopN):
                _offsets_into_windows(item, walker.bufs)
        return items, ()
    names = _actor_names(walked)
    shared = [item for actor, item in walked if actor is None]
    if shared:
        raise Refusal("THREAD_PARTITION_UNSUPPORTED", "work outside every role of a warp-specialized kernel")
    streams: "OrderedDict[str, List[Any]]" = OrderedDict()
    for actor, item in walked:
        streams.setdefault(names[actor], []).append(_tag(item, names[actor]))
    for role in list(streams):
        streams[role] = _reroll(streams[role], role, walker.bufs, walker.notes)
    loops = {role: [x for x in items if isinstance(x, LoopN)] for role, items in streams.items()}
    if any(len(v) > 1 for v in loops.values()):
        raise Refusal("WS_LOOP_STRUCTURE_UNSUPPORTED", "a role runs more than one loop: %s"
                      % {k: [l.name for l in v] for k, v in loops.items() if len(v) > 1})
    with_loop = [role for role, v in loops.items() if v]
    if not with_loop:
        return [x for items in streams.values() for x in items], tuple(streams)
    trips = {loops[role][0].trips for role in with_loop}
    if len(trips) != 1:
        raise Refusal("WS_LOOP_STRUCTURE_UNSUPPORTED", "role loops run different iteration counts %s" % sorted(trips))
    first = loops[with_loop[0]][0]
    merged = LoopN(first.name, first.trips, None)
    syncs: List[Op] = []
    for role, items in streams.items():
        position = next((i for i, x in enumerate(items) if isinstance(x, LoopN)), None)
        if position is None:
            merged.prologue.extend(items)
            continue
        loop = items[position]
        if loop.stages is not None:
            merged.stages = max(merged.stages or 0, loop.stages)
        merged.prologue.extend(items[:position])
        merged.body.extend(loop.body)
        merged.epilogue.extend(items[position + 1:])
        merged.issue_order.update(loop.issue_order)
        merged.meta.update(loop.meta)
        syncs.extend(getattr(loop, "_syncs", []))
        if loop.name != merged.name:
            _rename_axis(loop.body, loop.name, merged.name)
    body_ops = [op for op in merged.body if isinstance(op, Op)]
    if body_ops:
        low = min(op.window for op in body_ops)
        for op in body_ops:
            op.window -= low
    _pair_barriers(merged, syncs, walker.notes)
    rings = [walker.bufs[n].slots for op in merged.body if isinstance(op, Op) for n in _writes(op) if walker.bufs[n].slots]
    merged.stages = max([merged.stages or 0] + rings) or 1
    return [merged], tuple(streams)


def _pair_barriers(loop: LoopN, syncs: List[Op], notes: List[str]) -> None:
    """Each named-barrier generation pairs one arrive with one sync of the same barrier.

    Counting the events a role issues before its loop aligns the generations; the
    resulting ``after`` constraint releases the gated operation ``distance`` logical
    iterations after the arrive.
    """

    arrives = [op for op in loop.body if isinstance(op, Op) and op.kind == "barrier"]
    by_name = {op.name: op for op in loop.body if isinstance(op, Op)}
    ids = sorted({op.barrier[1] for op in arrives} | {op.barrier[1] for op in syncs})
    for barrier in ids:
        a = [op for op in arrives if op.barrier[1] == barrier]
        s = [op for op in syncs if op.barrier[1] == barrier]
        if len(a) != 1 or len(s) != 1 or a[0].actor == s[0].actor:
            raise Refusal("BARRIER_UNSUPPORTED", "named barrier %d: %d arrives and %d syncs per iteration"
                          % (barrier, len(a), len(s)))
        arrive, sync = a[0], s[0]
        meta_a, meta_s = loop.meta[arrive.actor], loop.meta[sync.actor]
        shift = (meta_a["pre"].get(("arrive", barrier), 0) - meta_a["start"]) \
            - (meta_s["pre"].get(("sync", barrier), 0) - meta_s["start"])
        gated = by_name[sync.output]
        distance = shift + gated.window - arrive.window
        if distance < 0:
            raise Refusal("BARRIER_UNSUPPORTED", "named barrier %d releases an earlier iteration" % barrier)
        loop.hints.append((arrive.name, gated.name, distance))
        notes.append("named barrier %d: %s waits for %s's arrive (%d iteration%s later)"
                     % (barrier, gated.name, arrive.actor, distance, "" if distance == 1 else "s"))


def _offsets_into_windows(loop: LoopN, bufs: Dict[str, Buf]) -> None:
    body = [op for op in loop.body if isinstance(op, Op)]
    if len(body) != len(loop.body):
        return
    offsets = _offsets(body, bufs)
    low = min(offsets) if offsets else 0
    for op, offset in zip(body, offsets):
        op.window = offset - low


def _mark_carried(items: List[Any], bufs: Dict[str, Buf]) -> None:
    """Register buffers whose value flows from one iteration of a loop into the next."""

    for item in items:
        if not isinstance(item, LoopN):
            continue
        _mark_carried(item.body, bufs)
        written = {n for op in item.body if isinstance(op, Op) for n in _writes(op)}
        seen: set = set()
        for op in item.body:
            if not isinstance(op, Op):
                continue
            for name in _reads(op):
                if name not in seen and name in written and bufs[name].storage == "register":
                    bufs[name].carried = True
            seen.update(_writes(op))


def _ssa(items: List[Any], bufs: Dict[str, Buf]) -> None:
    """Give every extra write of a non-carried register buffer in a loop body its own version."""

    for item in items:
        if not isinstance(item, LoopN):
            continue
        _ssa(item.body, bufs)
        current: Dict[str, str] = {}
        count: Dict[str, int] = {}
        for op in item.body:
            if not isinstance(op, Op):
                continue
            rename = lambda n: current.get(n, n)
            op.inputs = tuple(rename(n) for n in op.inputs)
            op.row_inputs = tuple(rename(n) for n in op.row_inputs)
            op.mapped_inputs = tuple(rename(n) for n in op.mapped_inputs)
            if isinstance(op.src, str):
                op.src = rename(op.src)
            for name in _writes(op):
                base = bufs[name]
                if base.carried or base.storage != "register":
                    continue
                count[name] = count.get(name, 0) + 1
                if count[name] == 1:
                    continue
                version = "%s__v%d" % (name, count[name] - 1)
                bufs.setdefault(version, replace(base, name=version))
                if op.output == name:
                    op.output = version
                elif op.dst == name:
                    op.dst = version
                current[name] = version


def _apply_splits(items: List[Any], splits: Dict[str, Tuple[str, str, int]]) -> None:
    """Picks recorded before a grid axis was split refer to both of its parts."""

    if not splits:
        return
    for op in items:
        if isinstance(op, LoopN):
            _apply_splits(op.body + op.prologue + op.epilogue, splits)
            continue
        for attr in ("src", "dst"):
            value = getattr(op, attr)
            if not isinstance(value, Slice):
                continue
            new_items = []
            for item in value.items:
                if item[0] == "tile" and item[1] in splits:
                    raise Refusal("INDEX_UNSUPPORTED", "%s tiles grid axis %s, which is also split" % (op.name, item[1]))
                if item[0] == "pick":
                    axes: List[str] = []
                    for axis in item[1]:
                        axes.extend((splits[axis][0], splits[axis][1]) if axis in splits else (axis,))
                    item = ("pick", tuple(axes))
                new_items.append(item)
            setattr(op, attr, Slice(value.tensor, tuple(new_items)))


def translate(capture: Capture, bindings: Mapping[str, Any]) -> Program:
    walker = _Walker(capture.planned, capture, bindings)
    walked = walker.walk()
    if walker.thread_extent is None:
        raise Refusal("LAUNCH_UNSUPPORTED", "no threadIdx.x binding in the planned kernel")
    if walker.thread_extent != capture.device.threads[0]:
        walker.notes.append("planned threadIdx.x=%d, device kernel threadIdx.x=%d; the device value is used"
                            % (walker.thread_extent, capture.device.threads[0]))
    items, actors = _structure(walked, walker)
    _apply_splits(items, walker.splits)
    _mark_carried(items, walker.bufs)
    _ssa(items, walker.bufs)
    order = {"blockIdx.x": 0, "blockIdx.y": 1, "blockIdx.z": 2}
    grid: "OrderedDict[str, int]" = OrderedDict()
    for _var, name, extent in sorted(walker.grid, key=lambda e: order.get(walker.grid_tags.get(e[0], ""), 9)):
        split = walker.splits.get(name)
        if split:
            hi, lo, factor = split
            grid[lo] = factor
            grid[hi] = extent // factor
        else:
            grid[name] = extent
    for name, tensor in walker.tensors.items():
        if tensor.view is not None:
            if tensor.view[0] == "flat":
                part = "all elements of %s as one dimension" % tensor.source
            else:
                part = "; ".join("dim %d: %s" % (d, "block %d of every %d" % (v[2], v[1]) if v[0] == "blocks"
                                                  else "elements %d..%d" % (v[1], v[1] + v[2] - 1))
                                 for d, v in enumerate(tensor.view[1]) if v is not None) + " of " + tensor.source
            walker.notes.append("%s declares %s (the same bytes, re-indexed; cache reuse with other views is not "
                                "modeled)" % (name, part))
    varying: List[LoopN] = []

    def collect(nodes: List[Any]) -> None:
        for node in nodes:
            if isinstance(node, LoopN):
                if isinstance(node.trips, CtaTrips):
                    varying.append(node)
                collect(node.prologue + node.body + node.epilogue)

    collect(items)
    groups, order_runs = walker.work_units(varying) if varying else ((), ())
    if varying:
        walker.notes.append("trip counts vary with the CTA (%s): %d work groups"
                            % (", ".join("%s = %s" % (l.name, l.trips) for l in varying), len(groups)))
    return Program(grid, capture.device.thread_count, walker.tensors, walker.bufs, items, actors,
                   walker.ws or bool(actors), walker.notes, walker.bindings, groups, order_runs)
