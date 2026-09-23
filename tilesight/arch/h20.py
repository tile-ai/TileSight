from tilesight.arch.arch_base import Arch

class H20(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "H100"
        self.sm_count = 78
        self.base_freq = 1.98 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.max_freq = 1.98 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4) # M,N.K
        self.tensor_core_flops = 256
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 4000 *1e9
        self.ddr_capacity = 141 * (1000**3)
        self.l2_bandwidth = 8123.448629e9 # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_capacity = 60 * (1024**2) # H100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 228 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 6
        # self.int8_int1_flops = self.fp16_tensor_flops * 12
        
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        # 111
        # self.ddr_max_util=1
        # self.l2_max_util= 0.93 
        # self.l1_max_util=1 
        # self.compute_max_util=1

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

    def set_to_spec(self):
        self

    def set_to_microbench(self):
        # 158.14656
        # -> 139.22
        self.base_freq = 1.743 * 1e9
        self.max_freq = 1.743 * 1e9
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0

        self.ddr_bandwidth = 4305 * 1e9
        self.l2_bandwidth= 8123.448629 *1e9
        # self.smem_bandwidth = 37699.2 * 1e9
        self.smem_bandwidth = 19768.320312 * 1e9


        self.fp32_cuda_core_flops = 29.231883 * 1e12
        self.fp16_cuda_core_flops = 33.323654 * 1e12
        self.fp64_cuda_core_flops = 0.617014 * 1e12
        self.fp16_tensor_flops = 139.22 * 1e12
        self.sfu_flops = 2.421514 * 1e12

        self.int8_tensor_flops = self.fp16_tensor_flops * 2
        self.fp8_tensor_flops = self.fp16_tensor_flops * 2

        return self

    def set_to_ncu(self):
        self.sm_count = 78
        # self.base_freq = 1.785 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        # self.max_freq = 1.785 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        # self.base_freq = 1.44 * 1e9 # 1.44 * 1e9 for tensorcore
        # self.max_freq = 1.44 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.base_freq = 1.65 * 1e9 # 1.44 * 1e9 for tensorcore
        self.max_freq = 1.65 * 1e9 # 1.83 * 1e9 #1.98 * 1e9
        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 4) # M,N.K
        self.tensor_core_flops = 256
        self.fp32_cores_per_sm = 128
        self.int32_cores_per_sm = 64
        self.ddr_bandwidth = 4000 *1e9
        self.ddr_capacity = 80 * (1024**3)
        self.l2_bandwidth = 8123.448629e9 # 96* self.max_freq * 2 * 32 # no compression, 2 sectors/cycle?, 95???
        self.l2_capacity = 60 * (1024**2) # H100 with dup
        self.sm_sub_partitions = 4
        self.l1_smem_throughput_per_cycle = 128
        self.configurable_smem_capacity = 228 * (1024**1)
        self.register_capacity_per_sm = 256 * (1024**1)
        self.warp_schedulers_per_sm = 4
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 6
        # self.int8_int1_flops = self.fp16_tensor_flops * 12
        
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 1
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.int32_cuda_core_flops = self.sm_count * self.max_freq * self.int32_cores_per_sm * 2
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        # 111
        # self.ddr_max_util=1
        # self.l2_max_util= 0.93 
        # self.l1_max_util=1 
        # self.compute_max_util=1

        self.ddr_max_util=0.85
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9


        return self

if __name__ == "__main__":
    arch = H20()
    arch.set_to_microbench()
    print("fp16_tensor_flops: ", arch.fp16_tensor_flops)
    print("int8_tensor_flops: ", arch.int8_tensor_flops)
    print("fp32_cuda_core_flops: ", arch.fp32_cuda_core_flops)
    print("fp16_cuda_core_flops: ", arch.fp16_cuda_core_flops)
    print("int32_cuda_core_flops: ", arch.int32_cuda_core_flops)
    print("sfu_flops: ", arch.sfu_flops)
    print("smem_bandwidth: ", arch.smem_bandwidth)
    print("register_bandwidth: ", arch.register_bandwidth)