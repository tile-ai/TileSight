# 文件名: p100.py
from .arch_base import Arch

class P100(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "P100"
        self.sm_count = 56
        self.base_freq = 1.328 * 1e9
        self.max_freq = 1.48 * 1e9
        self.tensor_cores_per_sm = 0  # P100 does not have Tensor Cores
        self.tensor_core_shape = (0, 0, 0)  # P100 does not have Tensor Cores, so shape is (0, 0, 0)
        self.tensor_core_flops = 0  # P100 does not have Tensor Core performance
        self.fp32_cores_per_sm = 64
        self.ddr_bandwidth = 732 * 1e9
        self.ddr_capacity = 16 * (1024**3)
        self.l2_bandwidth = 1624 * 1e9
        self.l2_capacity = 4 * (1024**2)
        self.sm_sub_partitions = 2
        self.l1_smem_throughput_per_cycle = 64
        self.configurable_smem_capacity = 64 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 2
        self.fp16_tensor_flops = 21.12 * 1e12
        self.fp32_cuda_core_flops = 10.62 * 1e12
        
        # Now calculate the derived properties
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

    # ... any other P100-specific methods or attributes
