from .arch_base import Arch

# =============================================================================
# AMD Instinct MI210 (CDNA2, gfx90a, single GCD)
#
# Three modes (mirrors h200_sxm.py):
#   __init__()           -> default STANDARD SPEC (derived peaks, util=0.9)
#   set_to_spec()        -> theoretical peak, util=1.0
#   set_to_microbench()  -> measured microbench values, util=1.0
#
# SPEC source: AMD CDNA2 / MI210 datasheet (the MI210 column):
#   Compute Units            104 CU
#   Stream processors        6,656            (= 104 * 64)
#   Matrix Cores             416              (= 104 * 4)
#   Peak FP64/FP32 Vector    22.6 TF
#   Peak FP64/FP32 Matrix    45.3 TF
#   Peak FP16/BF16           181.0 TF
#   Peak INT4/INT8           181.0 TOPS       (= FP16 rate, i.e. 1x)
#   Memory                   64 GB HBM2e, 4096-bit, 1.6 GHz, up to 1.6 TB/s
#   Max sclk                 1.7 GHz
#   Max power                300 W TDP, PCIe Gen4
#
# MI210 microbenchmark calibration.
#   (2026-05-25, real local MI210, Device 0x740f, 104 CU, ROCm 6.2.4 / gfx90a)
#
# NOTE: the old mi210.py used sm_count=110 + 1.316 GHz. Those came from an
#   MI250X/MI250 box (gfx90a reports 110 CU per GCD); the omniperf dump at the
#   bottom of this file is from that machine. The numbers here instead come from
#   a genuine MI210 (104 CU), so this file supersedes the old one.
# =============================================================================
class MI210(Arch):
    def __init__(self):
        super().__init__()  # base-class constructor

        self.core = "MI210"
        self.sm_count = 104                # CU count (MI210 datasheet & device query)
        self.base_freq = 1.7 * 1e9         # max sclk 1700 MHz
        self.max_freq = 1.7 * 1e9

        # Matrix cores (MFMA): 4 per CU. shape (8,4,4) -> 8*4*4*2 = 256 flops/core/cycle
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4)
        self.tensor_core_flops = 256

        # packed-FP32: model FP32 vector as 128 effective cores/CU (a cheap, common
        # optimization on CDNA2). 128 effective => 45.3 TF, matching the datasheet
        # "FP64/FP32 Matrix 45.3 TF" line. (Pure vector = 64 cores => 22.6 TF.)
        self.fp32_cores_per_sm = 128

        self.ddr_bandwidth = 1638.4 * 1e9  # 4096-bit @ 1.6 GHz DDR -> 1638.4 GB/s
        self.ddr_capacity = 64 * (1024**3) # 64 GB HBM2e

        # L2: 32 channels, CDNA2 doubled each slice to 128 B/clock.
        # 32 * 128 * 1.7 GHz = 6.96 TB/s theoretical peak (single GCD).
        self.l2_bandwidth = 32 * 128 * self.max_freq
        self.l2_capacity = 8 * (1024**2)   # 8 MiB

        self.sm_sub_partitions = 4         # 4 SIMD16 units per CU

        # LDS (shared memory): 32 banks * 4 B = 128 B/cycle/CU hardware peak.
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 64 * (1024**1)   # 64 KiB LDS per workgroup
        # MI2xx: 12.5 KiB SGPRs, 256 KiB VGPRs, 256 KiB AGPRs per CU
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 1    # kept from old model (CDNA2 has 4 SIMDs/CU)
        self.sfu_cores_per_sm = 16

        # ---- derived standard-spec peaks ----
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops  # 181.0 TF
        self.int8_tensor_flops = self.fp16_tensor_flops          # INT8/INT4 = FP16 rate (181 TOPS, 1x) on CDNA2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2          # 45.3 TF (packed)
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2      # 90.5 TF (packed fp16)
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5    # 22.6 TF (CDNA2 full-rate FP64)
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle          # 22.6 TB/s
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util = 0.9
        self.l2_max_util = 0.9
        self.l1_max_util = 0.9
        self.compute_max_util = 0.9

    def set_to_spec(self):
        """Theoretical peak (datasheet) with utilization = 1.0."""
        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        self.base_freq = 1.7 * 1e9
        self.max_freq = 1.7 * 1e9

        self.ddr_bandwidth = 1638.4 * 1e9
        self.l2_bandwidth = 32 * 128 * self.max_freq                                            # 6.96 TB/s
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle  # 22.6 TB/s (128 B/cycle)

        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops  # 181.0 TF
        self.int8_tensor_flops = self.fp16_tensor_flops                                          # 181 TOPS
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2          # 45.3 TF (packed)
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2      # 90.5 TF
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5    # 22.6 TF
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2

        return self

    def set_to_microbench(self):
        """Measured values from benchmark_results_on_mi210.sh (real MI210, 2026-05-25)."""
        # measured effective clock: 1679.7 MHz (nop kernel) / 1636.7 MHz (fp32-fma kernel)
        self.base_freq = 1.68 * 1e9
        self.max_freq = 1.68 * 1e9

        self.ddr_max_util = 1.0
        self.l2_max_util = 1.0
        self.l1_max_util = 1.0
        self.compute_max_util = 1.0

        # DRAM: v2 read/write/copy = 1512.7 / 1398.3 / 1415.7 GB/s (v1 read 1399.3).
        # Use sustained bidirectional (copy) as the model's effective HBM bandwidth.
        self.ddr_bandwidth = 1415.7 * 1e9

        # L2: dedicated test 5529.65 GB/s (sweep peak 5288.25 GB/s).
        self.l2_bandwidth = 4796.85 * 1e9

        # LDS/shared memory: measured whole-chip 13891.96 GB/s (theoretical 16972.8 GB/s).
        self.smem_bandwidth = 13891.96 * 1e9

        # CUDA-core (vector) throughput, measured TFLOPs.
        self.fp32_cuda_core_flops = 34.45 * 1e12   # FP32 FMAC min 34.45 / avg 34.44
        self.fp16_cuda_core_flops = 36.96 * 1e12   # FP16 FMAC 36.96
        self.fp64_cuda_core_flops = 10.04 * 1e12   # FP64 FMAC 10.04

        # MFMA fp16: use 4-wave/CU occupancy = 167.162 TF (the realistic GEMM
        # occupancy). Full-occupancy sweep peaks at 171.744 TF (needs 32 waves/CU);
        # simple bench 160.868 TF; end-to-end torch fp16 GEMM 16384^3 = 123.08 TF.
        self.fp16_tensor_flops = 167.16 * 1e12
        self.int8_tensor_flops = self.fp16_tensor_flops    # INT8 = FP16 rate on CDNA2

        # SFU: exp2f throughput = 1.081 TOPS
        self.sfu_flops = 1.081 * 1e12

        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        return self


if __name__ == "__main__":
    arch = MI210()
    print("== default standard spec ==")
    print("fp16_tensor_flops (TF): ", arch.fp16_tensor_flops / 1e12)
    print("int8_tensor_flops (TOPS):", arch.int8_tensor_flops / 1e12)
    print("fp32_cuda_core_flops(TF):", arch.fp32_cuda_core_flops / 1e12)
    print("fp16_cuda_core_flops(TF):", arch.fp16_cuda_core_flops / 1e12)
    print("fp64_cuda_core_flops(TF):", arch.fp64_cuda_core_flops / 1e12)
    print("l2_bandwidth (TB/s):     ", arch.l2_bandwidth / 1e12)
    print("smem_bandwidth (TB/s):   ", arch.smem_bandwidth / 1e12)
    print("ddr_bandwidth (TB/s):    ", arch.ddr_bandwidth / 1e12)

    arch.set_to_microbench()
    print("== microbench ==")
    print("fp16_tensor_flops (TF): ", arch.fp16_tensor_flops / 1e12)
    print("fp32_cuda_core_flops(TF):", arch.fp32_cuda_core_flops / 1e12)
    print("ddr_bandwidth (TB/s):    ", arch.ddr_bandwidth / 1e12)
    print("l2_bandwidth (TB/s):     ", arch.l2_bandwidth / 1e12)
    print("smem_bandwidth (TB/s):   ", arch.smem_bandwidth / 1e12)


# ============================================================================
# REFERENCE DUMP BELOW IS FROM AN MI250X/MI250 BOX (gfx90a, 110 CU per GCD),
# NOT the MI210 modeled above. Kept for archival / cross-checking only.
# ============================================================================

# Each GCD contains an L2 cache that is physically partitioned with one slice per memory controller and shared by all the resources on a
# single GCD. The AMD CDNA 2 family uses a 16-way set-associative design with 32 slices with a total capacity of 8MB (per GCD). To keep
# pace with the computational capabilities of the CUs, the bandwidth from each L2 slice has been doubled to 128B per clock - a peak of 6.96
# TB/s for the MI250, more than 2x the prior generation.

# *******
# Agent 8 (gfx90a, AMD Instinct MI250X/MI250)
# *******
#   Compute Unit:            110
#   SIMDs per CU:            4
#   Shader Engines:          8
#   Wavefront Size:          64
#   Max Waves Per CU:        32
#   Max Clock Freq. (MHz):   1700
#   Cache: L1 16 KB, L2 8192 KB
#   Total L2 Channels:       32
#   LDS Banks per CU:        32
#   HBM BW:                  1638.4 GB/s   (per GCD)
#   Num XCDs:                1
