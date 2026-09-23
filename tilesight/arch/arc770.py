from .arch_base import Arch
# Introduction to the Xe-HPG Architecture White Paper
# Intel Arc 770, 2.1GHz
# The core is Xe-HPG ACM-G10. 
# Whole ACM-G10 core has 8 Render Slices, 4Xe-Core per Render Slice, 
# 32 Xe Core (like SM) in total
# Each Xe-core wich 16 vecotre engines(XVE, 256 bit)
# Each XVE with 32 FP16 OPs/cycle; 16 FP32 OPs/cycle 
# Each Xe-core with 16 Matrix Engines(XMX, 1024 bit)
# Each XMX with 128 FP16/BF16 OPs/cycle(maybe 4-4-4), 256 INT8 OPs/cycle
# Each XVE/XMX shares 32KB Register File, means 256KB Register File per Xe-core
# 192 KB L1 Cache per Xe-core
# 128 KB Shared Local Memory per Xe-core
# 16 MB L2 Cache in total
# 32banks, 64bytes/cycle, 2048 bytes/cycle
# if 2.1GHz, L2 Bandwidth 2.1GHz*32banks*64bytes/cycle=4.3TB/s
# 560 GB/s DDR
# 137.6T Fp16
# 17.2T Fp32
# likely, 32 SM, 128 fp32 cudacore, 16 4-4-4 tensore core

class ARC770(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "MI210"
        self.sm_count = 32
        # self.base_freq = 1.4 * 1e9
        # self.max_freq = 1.4 * 1e9
        self.base_freq = 2.1 * 1e9
        self.max_freq = 2.1 * 1e9

        self.tensor_cores_per_sm = 16
        self.tensor_core_shape = (4, 4, 4)
        self.tensor_core_flops = 2048
        # self.fp32_cores_per_sm = 64
        self.fp32_cores_per_sm = 128 # 
        self.ddr_bandwidth = 560 * 1e9
        self.ddr_capacity = 16 * (1024**3)
        # 32banks, 64bytes/cycle, 2048 bytes/cycle
        self.l2_bandwidth = 4300.8 * 1e9
        self.l2_capacity = 16 * (1024**2) # 
        self.sm_sub_partitions = 16
        self.l1_smem_throughput_per_cycle = 128 # just guess=, not mentioned
        self.configurable_smem_capacity = 128 * (1024**1) # 192KB in total with L1$, just copied design like NV...
        self.register_capacity_per_sm = 256 * (1024**1) 
        self.warp_schedulers_per_sm = 16
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 17.2 * 1e12
        # self.fp32_cuda_core_flops = 137.63 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 4
        # self.int8_int1_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        # self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9    

        # by benchmark from https://github.com/chsasank/device-benchmarks
        self.fp32_cuda_core_flops = 15 * 1e12
        self.fp16_tensor_flops = 86 * 1e12
        self.ddr_bandwidth = 452 * 1e9    

        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9

# arch=ARC770()
# print(arch.fp16_tensor_flops/1e12)
# print(arch.smem_bandwidth/1e12)
# print(arch.fp32_cuda_core_flops/1e12)
