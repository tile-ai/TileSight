
from .arch_base import Arch

class Ampere_A6000(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "Ampere_A6000"
        self.sm_count = 84
        self.base_freq = 1.810 * 1e9
        self.max_freq = 1.80 * 1e9
        # self.base_freq = 1.41 * 1e9 #ncu
        # self.max_freq = 1.41 * 1e9 #ncu
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4)
        self.tensor_core_flops = 512
        self.fp32_cores_per_sm = 128
        self.ddr_bandwidth = 768 * 1e9
        self.ddr_capacity = 48 * (1024**3)
        self.l2_bandwidth = 2051.5 * 1e9
        self.l2_capacity = 6 * (1024**2) # 
        self.sm_sub_partitions = 4
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 100 * (1024**1) #128 in total
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        self.int4_tflops = self.fp16_tensor_flops * 4 
        # self.int8_int2_flops = self.fp16_tensor_flops * 6
        # self.int8_int1_flops = self.fp16_tensor_flops * 12
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        # self.ddr_max_util=0.9
        # self.l2_max_util=0.75
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

