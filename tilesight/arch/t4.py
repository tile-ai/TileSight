# 文件名: t4.py
from .arch_base import Arch

class T4(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments

        # After calling the base class constructor, set the properties
        self.core = "T100"
        self.sm_count = 40
        self.base_freq = 0.585 * 1e9
        self.max_freq = 1.59 * 1e9
        self.tensor_cores_per_sm = 8
        self.tensor_core_shape = (4, 4, 4)
        self.tensor_core_flops = 128
        self.fp32_cores_per_sm = 64
        self.ddr_bandwidth = 320 * 1e9
        self.ddr_capacity = 16 * (1024**3)
        self.l2_bandwidth = 1270 * 1e9
        self.l2_capacity = 4 * (1024**2)
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 64
        self.configurable_smem_capacity = 64 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.fp16_tensor_flops = 65.13 * 1e12
        self.fp32_cuda_core_flops = 8.14 * 1e12
        
        # Now calculate the derived properties
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

    # ... any other T4-specific methods or attributes
