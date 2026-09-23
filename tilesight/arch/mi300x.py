from .arch_base import Arch
# CDNA3 Whitepaper
# And check this
# https://github.com/nod-ai/shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md
class MI300X(Arch):
    def __init__(self):
        super().__init__()  # Call the base class constructor without arguments
        
        # After calling the base class constructor, set the properties
        self.core = "MI300X"
        self.sm_count = 304
        self.base_freq = 1.64 * 1e9
        self.max_freq = 1.64 * 1e9
        # omniperf: 10241348.5 cycles, 7779381.5ns, 1.316GHz
        # self.base_freq = 2.1 * 1e9
        # self.max_freq = 2.1 * 1e9

        self.tensor_cores_per_sm = 4
        self.tensor_core_shape = (8, 4, 8)
        self.tensor_core_flops = 512
        # self.fp32_cores_per_sm = 64
        self.fp32_cores_per_sm = 128 # considering packed fp32, a easy to use optimization, which makes fp32 cores double
        self.ddr_bandwidth = 5300 * 1e9
        self.ddr_capacity = 192 * (1024**3)
        # self.l2_bandwidth = 5288 * 1e9 ?????????????
        # The L2 cache is shared across the whole chip and physically partitioned into multiple slices. For the MI100, the cache is 16-way set
            # associative and comprises 32 slices (twice as many as in MI50) in total for an aggregate capacity of 8MB. Each slice can sustain 64B/cycle
            # for an aggregate bandwidth over 3TB/s across the GPU.
        self.l2_bandwidth = 10600 * 1e9
        self.l2_capacity = 4 * (1024**2) # effective cap here; A100 with dup
        # amd's wavefronts are 64 threads, 10 wavefronts per CU, so 640 threads per CU
        self.sm_sub_partitions = 4
        # self.l1_smem_throughput_per_cycle = 128 / 1.33
        
        # 64KB Local Data Share (LDS, or shared memory)
        # 16 KB Read/Write L1 vector data cache
        self.l1_smem_throughput_per_cycle = 128 # 110CUs*1700MHz*4byte*32bank=23.936TB/s, with info by omniperf
        # vL1D 11.968, sL1D 6.092, iL1D 6.092, TB/s
        self.configurable_smem_capacity = 64 * (1024**1)
        # MI2xx with: 12.5 KiB SGPRs, 256 KiB VGPRs, 256 KiB AGPRs per CU
        # GFX9 features large register files. Registers are DWORD-sized (4 B), and are split into 3 general groups:
        # SGPRs: Scalar registers (uniform value within subgroup threads). Up to 104 SGPRs per workgroup on MI300.
        # VGPRs: General-purpose vector registers (each thread holds a different value). Up to 256 VGPRs per thread on MI300.
        # AGPRs: Matrix accumulation vector registers (each thread holds a different value). Up to 256 AGPRs per thread on MI300.
        self.register_capacity_per_sm = 256 * (1024**1) ##
         
        self.warp_schedulers_per_sm = 1
        self.sfu_cores_per_sm  = 16
        # self.fp16_tensor_flops = 311.87 * 1e12
        # self.fp32_cuda_core_flops = 19.49 * 1e12
        
        # Now calculate the derived properties
        self.fp16_tensor_flops = self.sm_count * self.max_freq * self.tensor_cores_per_sm * self.tensor_core_flops
        # self.int8_tensor_flops = self.fp16_tensor_flops * 2 
        # self.int8_int2_flops = self.fp16_tensor_flops * 4
        # self.int8_int1_flops = self.fp16_tensor_flops * 2
        self.fp32_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2
        self.fp16_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 2
        self.fp64_cuda_core_flops = self.sm_count * self.max_freq * self.fp32_cores_per_sm * 2 * 0.5
        self.sfu_flops = self.sm_count * self.max_freq * self.sfu_cores_per_sm * 2 
        self.smem_bandwidth = self.sm_count * self.max_freq * self.l1_smem_throughput_per_cycle
        self.register_bandwidth = self.sm_count * self.max_freq * self.sm_sub_partitions * 32 * 4

        self.ddr_max_util=0.9
        self.l2_max_util=0.9
        self.l1_max_util=0.9
        self.compute_max_util=0.9

        # self.fp16_tensor_flops=168e12
        # self.ddr_bandwidth = 1406 * 1e9
        # self.smem_bandwidth  = 14691.22 * 1e9
        # # self.smem_bandwidth = 17556 * 1e9 by omniperf
        # self.fp32_cuda_core_flops = 35.245e12
        # self.fp16_cuda_core_flops = 35.245e12
        


        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9

    def set_to_microbench(self):
        self.base_freq = 1.64 * 1e9
        self.max_freq = 1.64 * 1e9
        # self.ddr_max_util=0.9
        # self.l2_max_util=0.9
        # self.l1_max_util=0.9
        # self.compute_max_util=0.9
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0

        self.ddr_bandwidth = 3816.48 * 1e9
        # self.l2_bandwidth= 16629.11 *1e9 
        self.l2_bandwidth= 10000 *1e9 
        # self.smem_bandwidth = 50160.04 * 1.64 / 2.1 * 1e9
        self.smem_bandwidth = 50160.04 * 1e9


        self.fp32_cuda_core_flops = 108.40 * 1e12
        # self.fp16_cuda_core_flops = 44.43 * 1e12
        # self.fp64_cuda_core_flops = 23.74 * 1e12
        self.fp16_cuda_core_flops = 108.40 * 1e12 * 2
        self.fp64_cuda_core_flops = 108.40 * 1e12 * 0.5
        self.fp16_tensor_flops = 962.498 * 1e12

        return self

    def set_to_spec(self):
        self.ddr_max_util=1.0
        self.l2_max_util=1.0
        self.l1_max_util=1.0
        self.compute_max_util=1.0
        self.base_freq = 2.1 * 1e9
        self.max_freq = 2.1 * 1e9
        
        return self



# arch=MI210()
# print(arch.fp16_tensor_flops/1e12)
# print(arch.smem_bandwidth/1e12)
# print(arch.fp32_cuda_core_flops/1e12)

# Each GCD contains an L2 cache that is physically partitioned with one slice per memory controller and shared by all the resources on a
# single GCD. The AMD CDNA 2 family uses a 16-way set-associative design with 32 slices with a total capacity of 8MB (per GCD). To keep
# pace with the computational capabilities of the CUs, the bandwidth from each L2 slice has been doubled to 128B per clock – a peak of 6.96
# TB/s for the MI250, more than 2x the prior generation 4
# . The queuing and arbitration for the distributed L2 cache have been enhanced to
# improve utilization of this read bandwidth over a wide range of workloads.

# *******                  
# Agent 10                 
# (3~10)
# *******                  
#   Name:                    gfx942                             
#   Uuid:                    GPU-d543c58ebef6cd61               
#   Marketing Name:          AMD Instinct MI300X VF             
#   Vendor Name:             AMD                                
#   Feature:                 KERNEL_DISPATCH                    
#   Profile:                 BASE_PROFILE                       
#   Float Round Mode:        NEAR                               
#   Max Queue Number:        128(0x80)                          
#   Queue Min Size:          64(0x40)                           
#   Queue Max Size:          131072(0x20000)                    
#   Queue Type:              MULTI                              
#   Node:                    9                                  
#   Device Type:             GPU                                
#   Cache Info:              
#     L1:                      32(0x20) KB                        
#     L2:                      4096(0x1000) KB                    
#     L3:                      262144(0x40000) KB                 
#   Chip ID:                 29877(0x74b5)                      
#   ASIC Revision:           1(0x1)                             
#   Cacheline Size:          64(0x40)                           
#   Max Clock Freq. (MHz):   2100                               
#   BDFID:                   0                                  
#   Internal Node ID:        9                                  
#   Compute Unit:            304                                
#   SIMDs per CU:            4                                  
#   Shader Engines:          32                                 
#   Shader Arrs. per Eng.:   1                                  
#   WatchPts on Addr. Ranges:4                                  
#   Coherent Host Access:    FALSE                              
#   Memory Properties:       
#   Features:                KERNEL_DISPATCH 
#   Fast F16 Operation:      TRUE                               
#   Wavefront Size:          64(0x40)                           
#   Workgroup Max Size:      1024(0x400)                        
#   Workgroup Max Size per Dimension:
#     x                        1024(0x400)                        
#     y                        1024(0x400)                        
#     z                        1024(0x400)                        
#   Max Waves Per CU:        32(0x20)                           
#   Max Work-item Per CU:    2048(0x800)                        
#   Grid Max Size:           4294967295(0xffffffff)             
#   Grid Max Size per Dimension:
#     x                        4294967295(0xffffffff)             
#     y                        4294967295(0xffffffff)             
#     z                        4294967295(0xffffffff)             
#   Max fbarriers/Workgrp:   32                                 
#   Packet Processor uCode:: 150                                
#   SDMA engine uCode::      21                                 
#   IOMMU Support::          None                               
#   Pool Info:               
#     Pool 1                   
#       Segment:                 GLOBAL; FLAGS: COARSE GRAINED      
#       Size:                    200753152(0xbf74000) KB            
#       Allocatable:             TRUE                               
#       Alloc Granule:           4KB                                
#       Alloc Recommended Granule:2048KB                             
#       Alloc Alignment:         4KB                                
#       Accessible by all:       FALSE                              
#     Pool 2                   
#       Segment:                 GLOBAL; FLAGS: EXTENDED FINE GRAINED
#       Size:                    200753152(0xbf74000) KB            
#       Allocatable:             TRUE                               
#       Alloc Granule:           4KB                                
#       Alloc Recommended Granule:2048KB                             
#       Alloc Alignment:         4KB                                
#       Accessible by all:       FALSE                              
#     Pool 3                   
#       Segment:                 GLOBAL; FLAGS: FINE GRAINED        
#       Size:                    200753152(0xbf74000) KB            
#       Allocatable:             TRUE                               
#       Alloc Granule:           4KB                                
#       Alloc Recommended Granule:2048KB                             
#       Alloc Alignment:         4KB                                
#       Accessible by all:       FALSE                              
#     Pool 4                   
#       Segment:                 GROUP                              
#       Size:                    64(0x40) KB                        
#       Allocatable:             FALSE                              
#       Alloc Granule:           0KB                                
#       Alloc Recommended Granule:0KB                                
#       Alloc Alignment:         0KB                                
#       Accessible by all:       FALSE                              
#   ISA Info:                
#     ISA 1                    
#       Name:                    amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-
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
# ====================================================================================
# rcu -s
# Machine Specifications: describing the state of the machine that ROCm Compute Profiler data was collected on.
# Output version: 3
# ╒════╤════════════════════════╤════════════════════════════════════════════════╤═════════════════════════════════════════════════════════════════════════════════════════════════════════╤════════╕
# │    │ Spec                   │ Value                                          │ Description                                                                                             │ Unit   │
# ╞════╪════════════════════════╪════════════════════════════════════════════════╪═════════════════════════════════════════════════════════════════════════════════════════════════════════╪════════╡
# │  0 │ Workload Name          │                                                │ The name of the workload data was collected for.                                                        │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  1 │ Command                │                                                │ The command the workload was executed with.                                                             │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  2 │ IP Blocks              │                                                │ The hardware blocks profiling information was collected for.                                            │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  3 │ Timestamp              │ Wed Apr  2 16:06:37 2025 (UTC)                 │ The time (in local system time) when data was collected                                                 │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  4 │ Hostname               │ node-0                                         │ The hostname of the machine.                                                                            │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  5 │ CPU Model              │ Intel(R) Xeon(R) Platinum 8480C                │ The model name of the CPU used.                                                                         │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  6 │ SBIOS                  │ Microsoft CorporationHyper-V UEFI Release v4.1 │ The system management bios version and vendor.                                                          │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  7 │ Linux Distribution     │ Ubuntu 22.04.5 LTS                             │ The Linux distribution installed on the machine.                                                        │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  8 │ Linux Kernel Version   │ 5.15.0-1073-azure                              │ The Linux kernel version running on the machine.                                                        │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │  9 │ AMD GPU Kernel Version │                                                │ [RESERVED] The version of the AMDGPU driver installed on the machine. Unimplemented.                    │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 10 │ CPU Memory             │ 1909357896                                     │ The total amount of memory available to the CPU.                                                        │ KB     │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 11 │ GPU Memory             │                                                │ [RESERVED] The total amount of memory available to accelerators/GPUs in the system. Unimplemented.      │ KB     │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 12 │ ROCm Version           │ 6.3.3-74                                       │ The ROCm version used during data-collection.                                                           │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 13 │ VBIOS                  │ 113-M3000100-101                               │ The version of the accelerators/GPUs video bios in the system.                                          │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 14 │ Compute Partition      │ NA                                             │ The compute partitioning mode active on the accelerators/GPUs in the system (MI300 only).               │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 15 │ Memory Partition       │ NPS1                                           │ The memory partitioning mode active on the accelerators/GPUs in the system (MI300 only).                │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 16 │ GPU Model              │ MI300                                          │ The product name of the accelerators/GPUs in the system.                                                │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 17 │ GPU Arch               │ gfx942                                         │ The architecture name of the accelerators/GPUs in the system,                                           │        │
# │    │                        │                                                │ as used by (e.g.,) the AMDGPU backed of LLVM.                                                           │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 18 │ GPU L1                 │ 32                                             │ The size of the vL1D cache (per compute-unit) on the accelerators/GPUs.                                 │ KiB    │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 19 │ GPU L2                 │ 4096                                           │ The size of the vL1D cache (per compute-unit) on the accelerators/GPUs.                                 │ KiB    │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 20 │ CU per GPU             │ 304                                            │ The total number of compute units per accelerator/GPU in the system. On systems with configurable       │        │
# │    │                        │                                                │ partitioning, (e.g., MI300) this is the total number of compute units in a partition.                   │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 21 │ SIMD per CU            │ 4                                              │ The number of SIMD processors in a compute unit for the accelerators/GPUs in the system.                │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 22 │ SE per GPU             │ 32                                             │ The number of shader engines on the accelerators/GPUs in the system. On systems with configurable       │        │
# │    │                        │                                                │ partitioning, (e.g., MI300) this is the total number of shader engines in a partition.                  │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 23 │ Wave Size              │ 64                                             │ The number work-items in a wavefront on the accelerators/GPUs in the system.                            │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 24 │ Workgroup Max Size     │ 1024                                           │ The maximum number of work-items in a workgroup on the accelerators/GPUs in the system.                 │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 25 │ Chip ID                │ 29877                                          │ <>                                                                                                      │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 26 │ Max Waves per CU       │ 32                                             │ The maximum number of wavefronts that can be resident on a compute unit on the                          │        │
# │    │                        │                                                │ accelerators/GPUs in the system                                                                         │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 27 │ Max SCLK               │ 2100                                           │ The maximum engine (compute-unit) clock rate of the accelerators/GPUs in the system.                    │ MHz    │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 28 │ Max MCLK               │ 1300                                           │ The maximum memory clock rate of the accelerators/GPUs in the system.                                   │ MHz    │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 29 │ Cur SCLK               │ 2100                                           │ [RESERVED] The current engine (compute unit) clock rate of the accelerators/GPUs in the system. Unused. │ MHz    │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 30 │ Cur MCLK               │ 1300                                           │ [RESERVED] The current memory clock rate of the accelerators/GPUs in the system. Unused.                │ MHz    │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 31 │ Total L2 Channels      │ 16                                             │ The maximum number of L2 cache channels on the accelerators/GPUs in the system. On systems with         │        │
# │    │                        │                                                │ configurable partitioning, (e.g., MI300) this is the total number of L2 cache channels in a partition.  │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 32 │ LDS Banks per CU       │ 32                                             │ The number of banks in the LDS for a compute unit on the accelerators/GPUs in the system.               │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 33 │ SQC per GPU            │ 160                                            │ The number of L1I/sL1D caches on the accelerators/GPUs in the system. On systems with                   │        │
# │    │                        │                                                │ configurable partitioning, (e.g., MI300) this is the total number of L1I/sL1D caches in a partition.    │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 34 │ Pipes per GPU          │ 4                                              │ The number of scheduler-pipes on the accelerators/GPUs in the system.                                   │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 35 │ HBM BW                 │ 665.6                                          │ The peak theoretical HBM bandwidth for the accelerators/GPUs in the system. On systems with             │ GB/s   │
# │    │                        │                                                │ configurable partitioning, (e.g., MI300) this is the peak theoretical HBM bandwidth for a partition.    │        │
# ├────┼────────────────────────┼────────────────────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
# │ 36 │ Num XCDs               │ 1                                              │ The total number of accelerator complex dies in a compute partition on the accelerators/GPUs in the     │ XCDs   │
# │    │                        │                                                │ system.  For accelerators without partitioning (i.e., pre-MI300), this is considered to be one.         │        │
# ╘════╧════════════════════════╧════════════════════════════════════════════════╧═════════════════════════════════════════════════════════════════════════════════════════════════════════╧════════╛
