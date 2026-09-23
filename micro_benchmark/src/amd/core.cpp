#define TILESIGHT_AMD_KERNELS
#include "common.hpp"
#include <chrono>

constexpr int THREADS = 256;
constexpr int ACCUMULATORS = 8;

__global__ void initialize_words(uint32_t* data, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) data[i] = payload_word(uint32_t(i));
}

__global__ void copy_words(const uint32_t* input, uint32_t* output, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) output[i] = input[i];
}

__global__ void read_words(const volatile uint32_t* input, uint32_t* output, size_t count, int iterations) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    uint32_t sum = 0;
    for (int repeat = 0; repeat < iterations; ++repeat)
        for (size_t i = tid; i < count; i += size_t(gridDim.x) * blockDim.x) sum += input[i];
    output[tid] = sum;
}

__device__ inline float multiply_add(float x, float y, float z) { return fmaf(x, y, z); }
__device__ inline double multiply_add(double x, double y, double z) { return fma(x, y, z); }

template<class T, bool Sqrt> __global__ void arithmetic(const T* input, T* output, int iterations) {
    const size_t tid = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    T accum[ACCUMULATORS];
    #pragma unroll
    for (int j = 0; j < ACCUMULATORS; ++j) accum[j] = input[tid * ACCUMULATORS + j];
    for (int i = 0; i < iterations; ++i) {
        #pragma unroll
        for (int j = 0; j < ACCUMULATORS; ++j) {
            if constexpr (Sqrt) accum[j] = sqrtf(accum[j] + T(0.25));
            else accum[j] = multiply_add(accum[j], T(0.999755859375), T(0.00018310546875));
        }
    }
    #pragma unroll
    for (int j = 0; j < ACCUMULATORS; ++j) output[tid * ACCUMULATORS + j] = accum[j];
}

__global__ void empty_kernel() {}

template<class T, bool Sqrt> void compute_probe(int blocks, int iterations, int repeats, const char* name) {
    const size_t count = size_t(blocks) * THREADS * ACCUMULATORS;
    Buffer<T> input(count), output(count);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(initialize_numeric<T>), dim3(blocks), dim3(THREADS), 0, 0,
                       input.data, count, 0x2468ace1u);
    auto launch = [&] { hipLaunchKernelGGL(HIP_KERNEL_NAME(arithmetic<T, Sqrt>), dim3(blocks), dim3(THREADS),
                                         0, 0, input.data, output.data, iterations); };
    launch();
    HIP_CHECK(hipDeviceSynchronize());
    for (int sample = 0; sample < repeats; ++sample) {
        const double ms = elapsed_ms(launch, 1);
        // FMA counts as two FLOPs. sqrt counts as one operation; its add is excluded.
        const double operations = double(count) * iterations * (Sqrt ? 1 : 2);
        metric(name, operations / (ms * (Sqrt ? 1e6 : 1e9)), Sqrt ? "Gop/s" : "TFLOP/s",
               blocks, sample, ms, 0, iterations);
    }
    std::vector<T> host(ACCUMULATORS);
    HIP_CHECK(hipMemcpy(host.data(), output.data, host.size() * sizeof(T), hipMemcpyDeviceToHost));
    for (int j = 0; j < ACCUMULATORS; ++j) {
        T expected = numeric_value<T>(j, 0x2468ace1u);
        for (int i = 0; i < iterations; ++i) {
            if constexpr (Sqrt) expected = std::sqrt(expected + T(0.25));
            else expected = std::fma(expected, T(0.999755859375), T(0.00018310546875));
        }
        if (!std::isfinite(host[j]) || std::abs(host[j] - expected) > std::abs(expected) * 1e-5)
            throw std::runtime_error("Arithmetic validation failed");
    }
}

int main(int argc, char** argv) {
    try {
        if (argc < 2) throw std::runtime_error("Expected a benchmark name");
        const std::string mode = argv[1];
        const auto properties = select_device(argc, argv);
        if (properties.maxThreadsPerBlock < THREADS || properties.multiProcessorCount < 1)
            throw std::runtime_error("Unsupported HIP device limits");
        const int default_blocks = properties.multiProcessorCount * 4;
        const int iterations = int(integer(option(argc, argv, "--iterations", "256"), INT32_MAX));
        const int repeats = int(integer(option(argc, argv, "--repeat", "3"), INT32_MAX));
        if (mode == "fp32") compute_probe<float, false>(default_blocks, iterations, repeats, "fp32_fma");
        else if (mode == "fp64") compute_probe<double, false>(default_blocks, iterations, repeats, "fp64_fma");
        else if (mode == "sfu") compute_probe<float, true>(default_blocks, iterations, repeats, "sqrt_expression");
        else if (mode == "launch") {
            hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, 0);
            HIP_CHECK(hipDeviceSynchronize());
            for (int sample = 0; sample < repeats; ++sample) {
                auto start = std::chrono::steady_clock::now();
                for (int i = 0; i < iterations; ++i) hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, 0);
                HIP_CHECK(hipGetLastError());
                HIP_CHECK(hipDeviceSynchronize());
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                metric("launch_batch_end_to_end", ms * 1000 / iterations, "us", 1, sample, ms, 0, iterations);
            }
        } else if (mode == "dram" || mode == "cache") {
            const size_t bytes = integer(option(argc, argv, "--bytes", "268435456"), uint64_t(SIZE_MAX) / 2);
            if (bytes < 4 || bytes % 4) throw std::runtime_error("--bytes must be a positive multiple of four");
            size_t free_bytes = 0, total_bytes = 0;
            HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
            if (2 * bytes > free_bytes * 0.8) throw std::runtime_error("Insufficient free device memory");
            const size_t count = bytes / sizeof(uint32_t);
            Buffer<uint32_t> input(count);
            hipLaunchKernelGGL(initialize_words, dim3(default_blocks), dim3(THREADS), 0, 0, input.data, count);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipDeviceSynchronize());
            if (mode == "dram") {
                Buffer<uint32_t> output(count);
                std::istringstream counts(option(argc, argv, "--blocks", std::to_string(default_blocks)));
                std::string item;
                while (std::getline(counts, item, ',')) {
                    const int blocks = int(integer(item, properties.maxGridSize[0]));
                    auto launch = [&] { hipLaunchKernelGGL(copy_words, dim3(blocks), dim3(THREADS), 0, 0,
                                                         input.data, output.data, count); };
                    launch();
                    HIP_CHECK(hipDeviceSynchronize());
                    for (int sample = 0; sample < repeats; ++sample) {
                        const double ms = elapsed_ms(launch, iterations);
                        metric("dram_copy_payload", 2.0 * bytes * iterations / (ms * 1e6), "GB/s",
                               blocks, sample, ms, bytes, iterations);
                    }
                    // Check beginning/middle/end outside the timing window.
                    for (size_t index : {size_t(0), count / 2, count - 1}) {
                        uint32_t actual = 0;
                        HIP_CHECK(hipMemcpy(&actual, output.data + index, sizeof(actual), hipMemcpyDeviceToHost));
                        if (actual != payload_word(uint32_t(index))) throw std::runtime_error("Copy validation failed");
                    }
                }
            } else {
                const int blocks = std::max(1, std::min(default_blocks, int(std::min<size_t>(count / THREADS, INT32_MAX))));
                Buffer<uint32_t> output(size_t(blocks) * THREADS);
                auto launch = [&] { hipLaunchKernelGGL(read_words, dim3(blocks), dim3(THREADS), 0, 0,
                                                      input.data, output.data, count, iterations); };
                launch();
                HIP_CHECK(hipDeviceSynchronize());
                for (int sample = 0; sample < repeats; ++sample) {
                    const double ms = elapsed_ms(launch, 1);
                    metric("cached_read_payload", double(bytes) * iterations / (ms * 1e6), "GB/s",
                           blocks, sample, ms, bytes, iterations);
                }
            }
        } else throw std::runtime_error("Unknown benchmark " + mode);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
