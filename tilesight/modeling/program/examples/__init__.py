"""Example programs, one module per kernel, all written with ``KernelBuilder``.

Run any of them with ``python -m tilesight.modeling.program.examples.<name>``; import the
builders with ``from tilesight.modeling.program import examples as ex``.

Every example is a runnable fixture for the public contract: no hand-written
``Timing``; service, extra latency (default zero, from the latency table when
given), completion, II and boundaries are computed by ``analyze``.  Each
docstring names the TileLang reference file and variant the structure follows
(local TileLang commit ``4c9cf5c7ea485e62312fb205c01cec68ad836075``); the
examples are modeling programs, not claims of equivalence with production
kernels, and accuracy is only what the validation entry has recorded.
"""

from importlib import import_module

_EXPORTS = {
    "TILELANG_COMMIT": "_common", "causal_kv_trips": "_common", "causal_valid_fraction": "_common",
    "causal_valid_fraction_rows": "_common", "report": "_common", "sectioned_lpt_order": "_common",
    "gemm_program": "gemm", "elementwise_add_program": "elementwise_add",
    "reduce_sum_program": "reductions", "rms_norm_program": "reductions",
    "fa3_program": "fa3", "fa4_program": "fa4_ts", "fa4_wasp_program": "fa4_wasp",
    "flashmla_decode_program": "flashmla_decode",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    """Resolve ``examples.<builder>`` on first use (keeps ``python -m ...examples.<module>`` a clean run)."""

    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    value = getattr(import_module("." + module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
