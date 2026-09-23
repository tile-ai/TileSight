#include <iostream>
#include "inline_ptx_func.hpp"
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/data_pattern.hpp"

#define SHARED_MEM_SIZE (8192 + 256)
#define WARP_SIZE 32

#ifndef THD_NUM
#define THD_NUM 128
#endif

#ifndef REP
#define REP 128
#endif

#ifndef TEST_MODE
#define TEST_MODE 0
#endif

// Add a new preprocessor definition for the number of iterations in bandwidth tests
#ifndef N_ITERS
#define N_ITERS 1000
#endif

static_assert(THD_NUM % WARP_SIZE == 0, "THD_NUM must be a multiple of WARP_SIZE");
static_assert(THD_NUM <= 256, "The 512-column allocation supports at most two warpgroups");
static_assert(REP <= 128, "Each TMEM stream occupies at most 128 columns");
static constexpr int kActiveWarps = THD_NUM / WARP_SIZE;
static constexpr double kBytesPerWarpTmemOp = 32.0 * REP * sizeof(uint32_t);
static constexpr double kBytesPerCtaTmemOp = kActiveWarps * kBytesPerWarpTmemOp;

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << ": " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

__device__ __forceinline__ unsigned long long read_clock64()
{
    unsigned long long value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory");
    return value;
}

// CUDA kernel
__global__ void benchmarkTMEM(unsigned long long *d_start, unsigned long long *d_end, uint32_t* data) {
    // Declare shared memory
    __shared__ uint32_t sharedMem[SHARED_MEM_SIZE];
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    if(warp_id == 0){
        tmem_allocate(sharedMem, 512);
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const uint32_t allocation_ptr = sharedMem[0];
    // A warp is restricted to its own 32-row quarter. Give each warpgroup
    // separate 256-column ranges so concurrent writers never overlap.
    uint32_t tmem_ptr = allocation_ptr + ((warp_id % 4) * 32u << 16)
                       + (warp_id / 4) * 256u;
    uint32_t tmem_ptr1 = tmem_ptr + 128;

    uint32_t val_array[REP];

    #pragma unroll
    for(int i = 0; i < REP; i++){
        val_array[i] = data[tid + i * THD_NUM];
    }

    // Pre-fill TMEM for read tests
    tmem_st_32dp32bNx<REP>(tmem_ptr, val_array);
    tmem_st_32dp32bNx<REP>(tmem_ptr1, val_array);
    fence_view_async_tmem_store();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    unsigned long long start[2];
    unsigned long long end[2];

#if TEST_MODE == 0 // Original Latency Test
    uint32_t val_array_tmp[REP];
    __syncthreads();
    __syncwarp();
    start[0] = read_clock64();

    tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
    fence_view_async_tmem_load();
    val_array[0] += val_array_tmp[0];

    end[0] = read_clock64();

#elif TEST_MODE == 6 // ================ WRITE BANDWIDTH TEST ================
    __syncthreads();
    __syncwarp();
    start[0] = read_clock64();

    // Loop many times to transfer a large amount of data
    // #pragma unroll 4
    #pragma unroll 4
    for (int i = 0; i < N_ITERS; ++i) {
        tmem_st_32dp32bNx<REP>(tmem_ptr, val_array);
    }
    // Wait for all stores to complete before stopping the clock
    fence_view_async_tmem_store();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    end[0] = read_clock64();

#elif TEST_MODE == 7 // ================ READ BANDWIDTH TEST =================
    uint32_t sink = 0; // Use this to "consume" the loaded data
    uint32_t val_array_tmp[REP];

    __syncthreads();
    __syncwarp();
    start[0] = read_clock64();

    // Loop many times to read a large amount of data
    #pragma unroll 4
    for (int i = 0; i < N_ITERS; ++i) {
        tmem_ld_32dp32bNx<REP>(tmem_ptr, val_array_tmp);
        // Complete the full register vector before the next iteration reuses it.
        fence_view_async_tmem_load();

        // Keep the load live without adding a large ALU reduction to the timed region.
        sink += val_array_tmp[0];
    }
    // Wait for the final load to complete
    fence_view_async_tmem_load();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    end[0] = read_clock64();

    // Also use the sink value to ensure it's not optimized away
    if (sink == 0xBADF00D) { // Condition will likely be false, but forces compiler to keep sink
        val_array[0] = sink;
    }
// 在内核的 #if/#elif 链的末尾添加
#elif TEST_MODE == 8 // ============ COMBINED READ + WRITE BANDWIDTH TEST ============
    uint32_t sink = 0; // "Sink" to consume loaded data and prevent optimization
    uint32_t val_array_tmp[REP];

    __syncthreads();
    __syncwarp();
    start[0] = read_clock64();

    // Loop many times to interleave read and write operations
    #pragma unroll 4
    for (int i = 0; i < N_ITERS; ++i) {
        // 1. Write to the first TMEM address
        tmem_st_32dp32bNx<REP>(tmem_ptr, val_array);

        // 2. Read from the second TMEM address
        tmem_ld_32dp32bNx<REP>(tmem_ptr1, val_array_tmp);
        fence_view_async_tmem_load();

        // 3. Keep the load live without adding a large ALU reduction to the timed region.
        sink += val_array_tmp[0];
    }

    // Wait for ALL operations (both store and load) to complete
    fence_view_async_tmem_store();
    fence_view_async_tmem_load();
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    end[0] = read_clock64();

    // Final use of the sink value to ensure it's not optimized away
    if (sink == 0xBADF00D) {
        val_array[0] = sink;
    }

#endif
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    if(warp_id == 0){
        tmem_free(allocation_ptr, 512);
    }
    __syncthreads();

    // Write the start and end times to global memory
    if (tid == 0) {
        d_start[0] = start[0];
        d_end[0] = end[0];
    }

    // Write back results to prevent dead code elimination
    #pragma unroll
    for(int i = 0; i < REP; i++){
        data[tid + i * THD_NUM] = val_array[i];
    }
}

int main(int argc, char** argv) {
    uint32_t seed = tilesight_bench::kDefaultDataSeed;
    if (argc != 1 && (argc != 3 || std::strcmp(argv[1], "--seed") ||
                      !tilesight_bench::parse_data_seed(argv[2], &seed))) {
        std::cerr << "Usage: tmem [--seed UNSIGNED_32_BIT_INTEGER]\n";
        return 2;
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (prop.major != 10 || prop.minor != 0) {
        std::cerr << "This TMEM probe requires SM100\n";
        return 2;
    }
    // Allocate memory on the device
    unsigned long long *d_start, *d_end;
    uint32_t *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(uint32_t) * THD_NUM * REP));
    CUDA_CHECK(cudaMalloc(&d_start, sizeof(unsigned long long) * 2));
    CUDA_CHECK(cudaMalloc(&d_end, sizeof(unsigned long long) * 2));
    CUDA_CHECK(cudaMemset(d_start, 0, sizeof(unsigned long long) * 2));
    CUDA_CHECK(cudaMemset(d_end, 0, sizeof(unsigned long long) * 2));
    std::vector<uint32_t> input(THD_NUM * REP);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = tilesight_bench::data_word(i, seed);
    CUDA_CHECK(cudaMemcpy(d_data, input.data(), input.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    std::cout << "data_pattern=index_hash_v1 data_seed=" << seed << '\n';

    // Get GPU clock rate for bandwidth calculation
    int deviceId;
    cudaGetDevice(&deviceId);
    int clockRate; // in kilohertz
    cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, deviceId);
    double gpu_clock_ghz = clockRate / 1000000.0;
    int smCount;
    CUDA_CHECK(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, deviceId));

    // Launch the kernel
    benchmarkTMEM<<<1, THD_NUM>>>(d_start, d_end, d_data);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy the start and end times back to the host
    unsigned long long h_start[2];
    unsigned long long h_end[2];
    CUDA_CHECK(cudaMemcpy(h_start, d_start, sizeof(unsigned long long) * 2, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_end, d_end, sizeof(unsigned long long) * 2, cudaMemcpyDeviceToHost));

    double duration_cycles = h_end[0] - h_start[0];
    if (duration_cycles <= 0.0) {
        std::cerr << "Invalid timing: start=" << h_start[0]
                  << ", end=" << h_end[0]
                  << ", cycles=" << duration_cycles << std::endl;
        std::exit(EXIT_FAILURE);
    }

    #if TEST_MODE == 0
        std::cout << "TMEM Load[0] Latency: " << duration_cycles << " clock cycles" << std::endl;
    #elif TEST_MODE == 1
        std::cout << "TMEM Store[0] + Load[0] Latency: " << duration_cycles << " clock cycles" << std::endl;
    // ... other original latency prints
    // #elif TEST_MODE == 6
    //     double bytes_written = (double)THD_NUM * N_ITERS * REP * sizeof(uint32_t);
    //     double duration_sec = duration_cycles / (gpu_clock_ghz * 1e9);
    //     double bandwidth_gbps = bytes_written / duration_sec / 1e9;
    //     std::cout << "TMEM Write Bandwidth (REP=" << REP << "): " << bandwidth_gbps << " GB/s" << std::endl;
    // #elif TEST_MODE == 7
    //     double bytes_read = (double)THD_NUM * N_ITERS * REP * sizeof(uint32_t);
    //     double duration_sec = duration_cycles / (gpu_clock_ghz * 1e9);
    //     double bandwidth_gbps = bytes_read / duration_sec / 1e9;
    //     std::cout << "TMEM Read Bandwidth (REP=" << REP << "): " << bandwidth_gbps << " GB/s" << std::endl;
    // #endif

    #elif TEST_MODE == 6 // MODIFIED FOR BYTES/CYCLE
        double bytes_written = kBytesPerCtaTmemOp * N_ITERS;
        double bytes_per_cycle = bytes_written / duration_cycles;
        double per_sm_gbps = bytes_per_cycle * gpu_clock_ghz;
        std::cout << "TMEM Write Throughput (per CTA/SM, REP=" << REP << "): "
                  << bytes_per_cycle << " Bytes/Cycle"
                  << " (" << per_sm_gbps << " GB/s per SM, "
                  << per_sm_gbps * smCount / 1000.0 << " TB/s whole-chip extrapolated)"
                  << std::endl;
        std::cout << "Raw timing: " << duration_cycles << " cycles, active warps: "
                  << kActiveWarps << ", bytes: " << bytes_written << std::endl;

    #elif TEST_MODE == 7 // MODIFIED FOR BYTES/CYCLE
        double bytes_read = kBytesPerCtaTmemOp * N_ITERS;
        double bytes_per_cycle = bytes_read / duration_cycles;
        double per_sm_gbps = bytes_per_cycle * gpu_clock_ghz;
        std::cout << "TMEM Read Throughput (per CTA/SM, REP=" << REP << "): "
                  << bytes_per_cycle << " Bytes/Cycle"
                  << " (" << per_sm_gbps << " GB/s per SM, "
                  << per_sm_gbps * smCount / 1000.0 << " TB/s whole-chip extrapolated)"
                  << std::endl;
        std::cout << "Raw timing: " << duration_cycles << " cycles, active warps: "
                  << kActiveWarps << ", bytes: " << bytes_read << std::endl;
    #elif TEST_MODE == 8 // MODIFIED FOR COMBINED R+W
        double total_bytes = kBytesPerCtaTmemOp * N_ITERS * 2.0;
        double bytes_per_cycle = total_bytes / duration_cycles;
        double per_sm_gbps = bytes_per_cycle * gpu_clock_ghz;
        std::cout << "TMEM Combined R+W Throughput (per CTA/SM, REP=" << REP << "): "
                  << bytes_per_cycle << " Bytes/Cycle"
                  << " (" << per_sm_gbps << " GB/s per SM, "
                  << per_sm_gbps * smCount / 1000.0 << " TB/s whole-chip extrapolated)"
                  << std::endl;
        std::cout << "Raw timing: " << duration_cycles << " cycles, active warps: "
                  << kActiveWarps << ", bytes: " << total_bytes << std::endl;
    #endif

    std::cout << "TILESIGHT_METRIC tmem_bytes_per_cycle_per_sm "
              << bytes_per_cycle << " bytes/cycle/SM\n";

    // Clean up
    CUDA_CHECK(cudaFree(d_start));
    CUDA_CHECK(cudaFree(d_end));
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
