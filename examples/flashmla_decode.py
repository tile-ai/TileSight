"""Dense FlashMLA decode with hardware-derived costs, including split/combine.

Run: python examples/flashmla_decode.py --num-splits 4
Use --num-splits 1 for one launch. No measurements or GPU are required.
"""

import argparse

import tilesight as sight
from tilesight.arch.h200_sxm import H200_SXM
from tilesight.modeling.program import examples as ex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-splits", type=int, default=4)
    args = parser.parse_args()
    program = ex.flashmla_decode_program(
        batch=4, heads=128, kv_len=1024, head_group=64,
        d_qk=576, d_v=512, block_n=64, num_splits=args.num_splits,
    )
    result = sight.analyze(
        program, H200_SXM().set_to_microbench(),
        options=sight.Options(cache="fast", ii_mode="periodic_best"),
    )
    print(f"Program total: {result.program.total_s * 1e6:.3f} us")
    for name, launch in result.launches.items():
        traffic = launch.traffic_total
        print(
            f"  {name}: body={launch.kernel_body_s * 1e6:.3f} us, "
            f"launch={launch.launch_overhead_s * 1e6:.3f} us, "
            f"DDR read={traffic.ddr_read_bytes:.0f} B, "
            f"DDR write={traffic.ddr_write_bytes:.0f} B"
        )
    print("load_KV breakdown (level, direction, cumulative bytes, launch share):")
    for row in result.ops["main/kv/load_KV"].breakdown(scope="launch"):
        print(" ", row)


if __name__ == "__main__":
    main()
