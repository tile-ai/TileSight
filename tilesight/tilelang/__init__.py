"""TileLang frontend: estimate a TileLang kernel's run time with TileSight.

The kernel is lowered once with TileLang's own pipeline; the program after
``PipelinePlanning`` (the compiler's warp-specialized roles, buffer stages and
copy engines already decided) is translated into ``tilesight.modeling.program``
and analyzed. TileLang is imported only when a kernel is profiled.

    from tilesight.tilelang import profile_source_file
    profile = profile_source_file("kernels/gemm.py", "build_kernel", factory_args={"m": 8192, "n": 8192, "k": 8192})
"""

from .api import Profile, dumps, profile_primfunc, profile_request, profile_source_file
from .entry import KernelRequest
from .errors import Refusal

__all__ = ["KernelRequest", "Profile", "Refusal", "dumps", "profile_primfunc", "profile_request", "profile_source_file"]
