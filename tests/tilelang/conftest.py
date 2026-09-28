"""TileLang frontend tests; those that lower a kernel need TileLang and skip without it."""

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
EXAMPLES = REPO / "examples" / "tilelang"


def _has_tilelang() -> bool:
    try:
        import tilelang
    except Exception:
        return False
    return bool(tilelang.__version__)


needs_tilelang = pytest.mark.skipif(not _has_tilelang(), reason="TileLang is not installed")


def example_request(name: str, **changes):
    from tilesight.tilelang import KernelRequest
    catalog = json.loads((EXAMPLES / "cases.json").read_text())
    case = catalog["cases"][name]
    fields = dict(file=str(EXAMPLES / case["file"]), function=case["function"], factory_args=case["factory_args"],
                  specialization=case["specialization"], target_arch=catalog["target_arch"],
                  pass_configs=case["pass_configs"], bindings=case["bindings"])
    fields.update(changes)
    return KernelRequest(**fields)


_PROFILES = {}


def example_profile(name: str):
    """One profile per example per session (lowering and analysis take seconds)."""

    if name not in _PROFILES:
        from tilesight.tilelang import profile_request
        _PROFILES[name] = profile_request(example_request(name))
    return _PROFILES[name]
