# 文件名: a100.py
from .arch_base import Arch
# https://www.techpowerup.com/gpu-specs/radeon-instinct-mi50.c3335
# Vega 20, GFX ID=gfx906 
class MI50(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "MI50"
        self.sm_count = 60
        self.base_freq = 1.20 * 1e9
        self.max_freq = 1.20 * 1e9
        # self.base_freq = 1.75 * 1e9
        # self.max_freq = 1.75 * 1e9

        # self.tensor_cores_per_sm = 4
        # self.tensor_core_shape = (8, 4, 8)
        # self.tensor_core_flops = 512
        self.fp32_cores_per_sm = 64
        self.ddr_bandwidth = 1024 * 1e9
        self.ddr_capacity = 16 * (1024**3)
        # self.l2_bandwidth = 5288 * 1e9 ?????????????
        # The L2 cache is shared across the whole chip and physically partitioned into multiple slices. For the MI100, the cache is 16-way set
            # associative and comprises 32 slices (twice as many as in MI50) in total for an aggregate capacity of 8MB. Each slice can sustain 64B/cycle
            # for an aggregate bandwidth over 3TB/s across the GPU.
        self.l2_bandwidth = 1500 * 1e9
        self.l2_capacity = 4 * (1024**2) # effective cap here; A100 with dup
        # amd's wavefronts are 64 threads, 10 wavefronts per CU, so 640 threads per CU
        self.sm_sub_partitions = 1
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        
        # 64KB Local Data Share (LDS, or shared memory)
        # 16 KB Read/Write L1 vector data cache
        self.l1_smem_throughput_per_cycle = 64 # just guess
        self.configurable_smem_capacity = 64 * (1024**1)
        self.register_capacity_per_sm = 128 * (1024**1) ## or 256?
        self.warp_schedulers_per_sm = 1
        # self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.fp16_tensor_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        # self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 4
        # self.int8_int1_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        # self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        # self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        # self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9



# Agent 3                  
# *******                  
#   Name:                    gfx906                             
#   Uuid:                    GPU-c9ac214172df888d               
#   Marketing Name:          AMD Radeon VII                     
#   Vendor Name:             AMD                                
#   Feature:                 KERNEL_DISPATCH                    
#   Profile:                 BASE_PROFILE                       
#   Float Round Mode:        NEAR                               
#   Max Queue Number:        128(0x80)                          
#   Queue Min Size:          64(0x40)                           
#   Queue Max Size:          131072(0x20000)                    
#   Queue Type:              MULTI                              
#   Node:                    2                                  
#   Device Type:             GPU                                
#   Cache Info:              
#     L1:                      16(0x10) KB                        
#     L2:                      8192(0x2000) KB                    
#   Chip ID:                 26287(0x66af)                      
#   ASIC Revision:           1(0x1)                             
#   Cacheline Size:          64(0x40)                           
#   Max Clock Freq. (MHz):   1801                               
#   BDFID:                   33792                              
#   Internal Node ID:        2                                  
#   Compute Unit:            60                                 
#   SIMDs per CU:            4                                  
#   Shader Engines:          4                                  
#   Shader Arrs. per Eng.:   1                                  
#   WatchPts on Addr. Ranges:4                                  
#   Coherent Host Access:    FALSE                              
#   Features:                KERNEL_DISPATCH 
#   Fast F16 Operation:      TRUE                               
#   Wavefront Size:          64(0x40)                           
#   Workgroup Max Size:      1024(0x400)                        
#   Workgroup Max Size per Dimension:
#     x                        1024(0x400)                        
#     y                        1024(0x400)                        
#     z                        1024(0x400)                        
#   Max Waves Per CU:        40(0x28)                           
#   Max Work-item Per CU:    2560(0xa00)                        
#   Grid Max Size:           4294967295(0xffffffff)             
#   Grid Max Size per Dimension:
#     x                        4294967295(0xffffffff)             
#     y                        4294967295(0xffffffff)             
#     z                        4294967295(0xffffffff)             
#   Max fbarriers/Workgrp:   32                                 
#   Packet Processor uCode:: 471                                
#   SDMA engine uCode::      145                                
#   IOMMU Support::          None                               
#   Pool Info:               
#     Pool 1                   
#       Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
#       Size:                    16760832(0xffc000) KB              
#       Allocatable:             TRUE                               
#       Alloc Granule:           4KB                                
#       Alloc Recommended Granule:2048KB                             
#       Alloc Alignment:         4KB                                
#       Accessible by all:       FALSE                              
#     Pool 2                   
#       Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
#       Size:                    16760832(0xffc000) KB              
#       Allocatable:             TRUE                               
#       Alloc Granule:           4KB                                
#       Alloc Recommended Granule:2048KB                             
#       Alloc Alignment:         4KB                                
#       Accessible by all:       FALSE                              
#     Pool 3                   
#       Segment:                 GROUP                              
#       Size:                    64(0x40) KB                        
#       Allocatable:             FALSE                              
#       Alloc Granule:           0KB                                
#       Alloc Recommended Granule:0KB                                
#       Alloc Alignment:         0KB                                
#       Accessible by all:       FALSE                              
#   ISA Info:                
#     ISA 1                    
#       Name:                    amdgcn-amd-amdhsa--gfx906:sramecc+:xnack-
#       Machine Models:          HSA_MACHINE_MODEL_LARGE            
#       Profiles:                HSA_PROFILE_BASE                   
#       Default Rounding Mode:   NEAR                               
#       Default Rounding Mode:   NEAR                               
#       Fast f16:                TRUE                               
#       Workgroup Max Size:      1024(0x400)                        
#       Workgroup Max Size per Dimension:
#         x                        1024(0x400)                        
#         y                        1024(0x400)                        
#         z                        1024(0x400)                        
#       Grid Max Size:           4294967295(0xffffffff)             
#       Grid Max Size per Dimension:
#         x                        4294967295(0xffffffff)             
#         y                        4294967295(0xffffffff)             
#         z                        4294967295(0xffffffff)             
#       FBarrier Max Size:       32                                 
# *** Done ***             