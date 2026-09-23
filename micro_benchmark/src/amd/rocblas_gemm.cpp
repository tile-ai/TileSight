#define TILESIGHT_AMD_KERNELS
#include "common.hpp"
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

inline void blas_check(rocblas_status status) {
    if (status != rocblas_status_success) throw std::runtime_error("rocBLAS status " + std::to_string(int(status)));
}

__global__ void initialize_half(__half* data, size_t count, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] = __float2half(numeric_value<float>(i, seed));
}

inline float half_input(size_t index, uint32_t seed) {
    return __half2float(__float2half(numeric_value<float>(index, seed)));
}

int main(int argc, char** argv) {
    try {
        const auto properties = select_device(argc, argv);
        const int n = int(integer(option(argc, argv, "--size", "4096"), 32768));
        const int repeats = int(integer(option(argc, argv, "--repeat", "7"), INT32_MAX));
        const size_t count = size_t(n) * n;
        size_t free_bytes = 0, total_bytes = 0;
        HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
        if (count * (2 * sizeof(__half) + sizeof(float)) > free_bytes * 0.8)
            throw std::runtime_error("Insufficient free device memory");
        Buffer<__half> a(count), b(count);
        Buffer<float> c(count);
        const int blocks = properties.multiProcessorCount * 4;
        hipLaunchKernelGGL(initialize_half, dim3(blocks), dim3(256), 0, 0,
                           a.data, count, 0x12345678u);
        hipLaunchKernelGGL(initialize_half, dim3(blocks), dim3(256), 0, 0,
                           b.data, count, 0x87654321u);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
        rocblas_handle handle = nullptr;
        blas_check(rocblas_create_handle(&handle));
        blas_check(rocblas_set_pointer_mode(handle, rocblas_pointer_mode_host));
        const float alpha = 1, beta = 0;
        // FP16 inputs, FP32 accumulation/output. rocBLAS selects the implementation;
        // the result is library GEMM throughput, not a raw MFMA issue measurement.
        auto launch = [&] { blas_check(rocblas_gemm_ex(
            handle, rocblas_operation_none, rocblas_operation_none, n, n, n,
            &alpha, a.data, rocblas_datatype_f16_r, n, b.data, rocblas_datatype_f16_r, n,
            &beta, c.data, rocblas_datatype_f32_r, n, c.data, rocblas_datatype_f32_r, n,
            rocblas_datatype_f32_r, rocblas_gemm_algo_standard, 0, 0)); };
        launch();
        launch();
        HIP_CHECK(hipDeviceSynchronize());
        for (int sample = 0; sample < repeats; ++sample) {
            const double ms = elapsed_ms(launch, 1);
            metric("rocblas_fp16_f32acc_gemm", 2.0 * n * n * n / (ms * 1e9), "TFLOP/s", 0, sample, ms, 0, 1);
        }
        // Independently reconstruct three column-major output elements outside timing.
        for (size_t index : {size_t(0), count / 2, count - 1}) {
            const size_t row = index % n, column = index / n;
            double expected = 0;
            for (int k = 0; k < n; ++k)
                expected += double(half_input(row + size_t(k) * n, 0x12345678u))
                            * half_input(k + column * n, 0x87654321u);
            float actual = 0;
            HIP_CHECK(hipMemcpy(&actual, c.data + index, sizeof(actual), hipMemcpyDeviceToHost));
            if (!std::isfinite(actual) || std::abs(actual - expected) > std::abs(expected) * 1e-3)
                throw std::runtime_error("rocBLAS GEMM validation failed");
        }
        blas_check(rocblas_destroy_handle(handle));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
