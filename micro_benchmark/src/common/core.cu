// Shared calibration probes. All bandwidths use decimal bytes/s.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include "data_pattern.hpp"

#define CHECK(call) do { auto e = (call); if (e != cudaSuccess) { \
    std::cerr << #call << ": " << cudaGetErrorString(e) << '\n'; std::exit(1); } } while (0)

static void metric(const char* name, double value, const char* unit) {
    if (!std::isfinite(value) || value <= 0) {
        std::cerr << "Invalid measurement: " << name << '\n'; std::exit(1);
    }
    std::cout << "TILESIGHT_METRIC " << name << ' ' << std::setprecision(12)
              << value << ' ' << unit << '\n';
}

template<class Launch> static double timed(Launch launch, int repeat) {
    for (int i = 0; i < 3; ++i) launch();
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));
    std::vector<double> seconds;
    for (int i = 0; i < repeat; ++i) {
        CHECK(cudaEventRecord(start)); launch(); CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop)); CHECK(cudaGetLastError());
        float ms = 0;
        CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (ms <= 0) { std::cerr << "Timer resolution exceeded\n"; std::exit(1); }
        seconds.push_back(ms * 1e-3);
    }
    CHECK(cudaEventDestroy(start)); CHECK(cudaEventDestroy(stop));
    std::sort(seconds.begin(), seconds.end());
    return seconds[seconds.size() / 2];
}

template<int MODE> __global__ void arithmetic(float* out, int iters) {
    float f[8]; double d[8]; __half2 h[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        f[j] = 0.25f + j * 0.01f; d[j] = f[j]; h[j] = __float2half2_rn(f[j]);
    }
    const __half2 a = __float2half2_rn(0.999f), b = __float2half2_rn(0.001f);
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            if constexpr (MODE == 0)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f[j]) : "f"(1.000001f), "f"(0.001f));
            if constexpr (MODE == 1) h[j] = __hfma2(h[j], a, b);
            if constexpr (MODE == 2)
                asm volatile("fma.rn.f64 %0, %0, %1, %2;" : "+d"(d[j]) : "d"(1.000001), "d"(0.001));
            if constexpr (MODE == 3)
                asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(f[j]) : "f"(-f[j]));
        }
    }
    float sum = 0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        if constexpr (MODE == 1) sum += __low2float(h[j]) + __high2float(h[j]);
        else if constexpr (MODE == 2) sum += float(d[j]);
        else sum += f[j];
    }
    out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__device__ __forceinline__ uint4 load_cg(const uint4* ptr) {
    uint4 v;
    asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
        : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(ptr) : "memory");
    return v;
}

__host__ __device__ static uint4 payload(size_t index, uint32_t seed) {
    return make_uint4(tilesight_bench::data_word(uint64_t(index) * 4, seed),
                      tilesight_bench::data_word(uint64_t(index) * 4 + 1, seed),
                      tilesight_bench::data_word(uint64_t(index) * 4 + 2, seed),
                      tilesight_bench::data_word(uint64_t(index) * 4 + 3, seed));
}

__global__ void initialize_payload(uint4* data, size_t count, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] = payload(i, seed);
}

__global__ void l2_read(const uint4* data, uint4* sink, size_t count, int iters) {
    size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    uint4 sum = make_uint4(0, 0, 0, 0);
    for (int i = 0; i < iters; ++i) {
        uint4 v = load_cg(data + (tid + size_t(i) * stride) % count);
        sum.x += v.x; sum.y += v.y; sum.z += v.z; sum.w += v.w;
    }
    sink[tid] = sum;
}

__global__ void chase(const unsigned* data, unsigned* sink, unsigned long long* cycles, int iters) {
    unsigned idx = 0;
    unsigned long long start = clock64();
    for (int i = 0; i < iters; ++i)
        asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(data + idx) : "memory");
    *cycles = clock64() - start;
    *sink = idx;
}

__global__ void shared_read(unsigned long long* cycles, unsigned* sink, int iters, uint32_t seed) {
    __shared__ uint4 data[256];
    data[threadIdx.x] = payload(threadIdx.x, seed);
    __syncthreads();
    unsigned address = unsigned(__cvta_generic_to_shared(data + threadIdx.x));
    uint4 v; unsigned sum = 0;
    unsigned long long start = clock64();
    for (int i = 0; i < iters; ++i) {
        asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
            : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(address) : "memory");
        sum += v.x + v.y + v.z + v.w;
    }
    __syncthreads();
    if (threadIdx.x == 0) *cycles = clock64() - start;
    sink[threadIdx.x] = sum;
}

__global__ void measure_clock(unsigned long long* result, float* sink, int iters) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0) :: "memory");
    auto c0 = clock64();
    float value = 0.25f;
    for (int i = 0; i < iters; ++i)
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(value) : "f"(1.000001f), "f"(0.001f));
    auto c1 = clock64();
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1) :: "memory");
    if (threadIdx.x == 0) { result[0] = c1 - c0; result[1] = t1 - t0; *sink = value; }
}

__global__ void empty_kernel() {}

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "Usage: core MODE [--iters N] [--repeat N] [--seed 1729]\n"; return 2; }
    std::string mode = argv[1];
    int iters = 4096, repeat = 5;
    uint32_t seed = tilesight_bench::kDefaultDataSeed;
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (++i >= argc) return 2;
        if (key == "--seed") {
            if (!tilesight_bench::parse_data_seed(argv[i], &seed)) return 2;
            continue;
        }
        char* end = nullptr; long v = std::strtol(argv[i], &end, 10);
        if (!*argv[i] || *end || v <= 0 || v > 10000000) return 2;
        if (key == "--iters") iters = int(v);
        else if (key == "--repeat") repeat = int(v);
        else return 2;
    }
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{}; CHECK(cudaGetDeviceProperties(&prop, 0));
    const int threads = 256, blocks = prop.multiProcessorCount * 4;
    size_t workers = size_t(threads) * blocks;
    float* sink; CHECK(cudaMalloc(&sink, workers * sizeof(uint4)));
    unsigned long long* clocks; CHECK(cudaMalloc(&clocks, 2 * sizeof(unsigned long long)));
    std::cout << "mode=" << mode << " iters=" << iters << " repeat=" << repeat
              << " blocks=" << blocks << " threads=" << threads << '\n';
    if (mode == "l2" || mode == "smem")
        std::cout << "data_pattern=index_hash_v1 data_seed=" << seed << '\n';
    if (mode == "fp32" || mode == "fp16" || mode == "fp64" || mode == "sfu") {
        auto launch = [&] {
            if (mode == "fp32") arithmetic<0><<<blocks, threads>>>(sink, iters);
            if (mode == "fp16") arithmetic<1><<<blocks, threads>>>(sink, iters);
            if (mode == "fp64") arithmetic<2><<<blocks, threads>>>(sink, iters);
            if (mode == "sfu") arithmetic<3><<<blocks, threads>>>(sink, iters);
        };
        double seconds = timed(launch, repeat);
        float value; CHECK(cudaMemcpy(&value, sink, sizeof(float), cudaMemcpyDeviceToHost));
        if (!std::isfinite(value)) { std::cerr << "Non-finite arithmetic output\n"; return 1; }
        double per_op = mode == "fp16" ? 4 : mode == "sfu" ? 1 : 2;
        std::string name = mode == "sfu" ? "sfu_exp2_tops" : mode + "_cuda_tflops";
        metric(name.c_str(), double(workers) * iters * 8 * per_op / seconds / 1e12,
               mode == "sfu" ? "TOP/s" : "TFLOP/s");
    } else if (mode == "l2") {
        if (prop.l2CacheSize <= 0) return 1;
        size_t bytes = (size_t(prop.l2CacheSize) / 2 / sizeof(uint4)) * sizeof(uint4);
        uint4* data; CHECK(cudaMalloc(&data, bytes));
        initialize_payload<<<blocks, threads>>>(data, bytes / sizeof(uint4), seed);
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        double seconds = timed([&] { l2_read<<<blocks, threads>>>(data, reinterpret_cast<uint4*>(sink), bytes / sizeof(uint4), iters); }, repeat);
        for (size_t sample : {size_t(0), workers / 2, workers - 1}) {
            uint4 actual, expected = make_uint4(0, 0, 0, 0);
            CHECK(cudaMemcpy(&actual, reinterpret_cast<uint4*>(sink) + sample, sizeof(actual), cudaMemcpyDeviceToHost));
            for (int i = 0; i < iters; ++i) {
                uint4 value = payload((sample + size_t(i) * workers) % (bytes / sizeof(uint4)), seed);
                expected.x += value.x; expected.y += value.y; expected.z += value.z; expected.w += value.w;
            }
            if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
                std::cerr << "L2 payload validation failed\n"; return 1;
            }
        }
        metric("l2_working_set", double(bytes), "bytes");
        metric("l2_read_gbps", double(workers) * iters * sizeof(uint4) / seconds / 1e9, "GB/s");
        CHECK(cudaFree(data));
    } else if (mode == "l2_latency") {
        size_t bytes = std::min(size_t(1 << 20), size_t(prop.l2CacheSize) / 4);
        bytes = bytes / 128 * 128;
        if (bytes < 128) return 1;
        std::vector<unsigned> next(bytes / sizeof(unsigned));
        for (size_t i = 0; i < next.size(); ++i) next[i] = unsigned((i + 32) % next.size());
        unsigned* data; CHECK(cudaMalloc(&data, bytes));
        CHECK(cudaMemcpy(data, next.data(), bytes, cudaMemcpyHostToDevice));
        chase<<<1, 1>>>(data, reinterpret_cast<unsigned*>(sink), clocks, int(next.size() / 32));
        CHECK(cudaDeviceSynchronize());
        std::vector<double> samples;
        for (int r = 0; r < repeat; ++r) {
            chase<<<1, 1>>>(data, reinterpret_cast<unsigned*>(sink), clocks, iters);
            CHECK(cudaGetLastError());
            unsigned long long cycles; CHECK(cudaMemcpy(&cycles, clocks, sizeof(cycles), cudaMemcpyDeviceToHost));
            samples.push_back(double(cycles) / iters);
        }
        std::sort(samples.begin(), samples.end());
        metric("l2_dependent_load_cycles", samples[samples.size() / 2], "cycles/load");
        CHECK(cudaFree(data));
    } else if (mode == "smem" || mode == "clock") {
        std::vector<double> samples;
        for (int r = 0; r < repeat + 2; ++r) {
            if (mode == "smem") shared_read<<<1, threads>>>(clocks, reinterpret_cast<unsigned*>(sink), iters, seed);
            else measure_clock<<<1, 32>>>(clocks, sink, std::max(iters, 100000));
            CHECK(cudaGetLastError());
            unsigned long long values[2]; CHECK(cudaMemcpy(values, clocks, sizeof(values), cudaMemcpyDeviceToHost));
            if (r >= 2) samples.push_back(mode == "smem" ? double(threads) * iters * sizeof(uint4) / values[0] : double(values[0]) / values[1]);
        }
        std::sort(samples.begin(), samples.end());
        if (mode == "smem") {
            for (unsigned sample : {0u, unsigned(threads / 2), unsigned(threads - 1)}) {
                uint4 value = payload(sample, seed);
                unsigned actual, expected = (value.x + value.y + value.z + value.w) * unsigned(iters);
                CHECK(cudaMemcpy(&actual, reinterpret_cast<unsigned*>(sink) + sample, sizeof(actual), cudaMemcpyDeviceToHost));
                if (actual != expected) { std::cerr << "Shared-memory payload validation failed\n"; return 1; }
            }
        }
        metric(mode == "smem" ? "shared_read_bytes_per_cycle_per_sm" : "sm_clock_ghz_single_cta",
               samples[samples.size() / 2], mode == "smem" ? "bytes/cycle/SM" : "GHz");
    } else if (mode == "launch") {
        for (int i = 0; i < 100; ++i) empty_kernel<<<1, 1>>>();
        CHECK(cudaDeviceSynchronize());
        auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) empty_kernel<<<1, 1>>>();
        auto end = std::chrono::steady_clock::now();
        CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
        metric("host_submit_us", std::chrono::duration<double>(end - begin).count() * 1e6 / iters, "us/launch");
        double seconds = timed([&] { for (int i = 0; i < iters; ++i) empty_kernel<<<1, 1>>>(); }, repeat);
        metric("empty_kernel_event_us", seconds * 1e6 / iters, "us/launch");
    } else { std::cerr << "Unknown benchmark mode\n"; return 2; }
    CHECK(cudaFree(clocks)); CHECK(cudaFree(sink));
}
