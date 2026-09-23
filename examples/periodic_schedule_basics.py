"""Compare II modes with a tile program and hardware-derived costs (CPU only).

Run after installing TileSight: python examples/periodic_schedule_basics.py
"""

import tilesight as sight
from tilesight.arch.h200_sxm import H200_SXM
from tilesight.modeling.program import examples as ex


def main():
    arch = H200_SXM().set_to_microbench()
    for stages in (1, 2, 3):
        program = ex.gemm_program(
            m=1024, n=1024, k=1024, stages=stages, dtype="bf16",
        )
        print(f"GEMM, shared-memory stages={stages}")
        for mode in ("resource_ii", "periodic_best", "periodic_worst"):
            result = sight.analyze(
                program, arch, options=sight.Options(cache="fast", ii_mode=mode),
            )
            ii = result.regions["main/k"].groups[0].ii
            launch = result.launches["main"]
            print(
                f"  {mode}: II={ii.selected_ii_s * 1e9:.2f} ns, "
                f"body={launch.kernel_body_s * 1e6:.2f} us, "
                f"witness={ii.witness is not None}, "
                f"search_complete={ii.search_complete}"
            )
            print(
                f"    bounds (ns): resource={ii.resource_ii_s * 1e9:.2f}, "
                f"recurrence={ii.recurrence_ii_s * 1e9:.2f}, "
                f"credit={ii.credit_ii_s * 1e9:.2f}; scope={ii.scope}"
            )


if __name__ == "__main__":
    main()
