"""FA3/H200 and FA4/B200 through KernelBuilder programs and analyze (CPU only).

Run: python examples/fa3_fa4_schedule_modes.py
Use --resource-only to skip periodic search. No per-op timing is supplied.
"""

import argparse

import tilesight as sight
from tilesight.arch.b200 import B200
from tilesight.arch.h200_sxm import H200_SXM
from tilesight.modeling.program import examples as ex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resource-only", action="store_true")
    args = parser.parse_args()
    shape = dict(batch=1, heads=8, seq_len=512, head_dim=64, causal=True)
    cases = (
        ("FA3", ex.fa3_program(**shape), H200_SXM().set_to_microbench()),
        ("FA4 ts", ex.fa4_program(**shape), B200().set_to_microbench()),
    )
    modes = ("resource_ii",) if args.resource_only else (
        "resource_ii", "periodic_best", "periodic_worst",
    )
    for name, program, arch in cases:
        for mode in modes:
            result = sight.analyze(
                program, arch, options=sight.Options(cache="fast", ii_mode=mode),
            )
            launch = result.launches["main"]
            print(
                f"{name} {mode}: body={launch.kernel_body_s * 1e6:.3f} us, "
                f"launch={launch.launch_overhead_s * 1e6:.3f} us, "
                f"host={launch.host_dispatch_s * 1e6:.3f} us"
            )
            # Causal Q tiles have different KV trip counts: retain every group.
            for group in result.regions["main/kv"].groups:
                ii = group.ii
                print(
                    f"  {group.work_group_id}, trips={group.trip_count}: "
                    f"II={ii.selected_ii_s * 1e9:.3f} ns, "
                    f"resource={ii.resource_ii_s * 1e9:.3f} ns, "
                    f"witness={ii.witness is not None}, "
                    f"search_complete={ii.search_complete}, scope={ii.scope}"
                )


if __name__ == "__main__":
    main()
