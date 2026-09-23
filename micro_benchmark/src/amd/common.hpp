#pragma once
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

inline void hip_check(hipError_t error, const char* call) {
    if (error != hipSuccess) throw std::runtime_error(std::string(call) + ": " + hipGetErrorString(error));
}
#define HIP_CHECK(call) hip_check((call), #call)

inline std::string option(int argc, char** argv, const std::string& key, const std::string& fallback) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] == key) {
            if (i + 1 == argc) throw std::runtime_error("Missing value for " + key);
            return argv[i + 1];
        }
    }
    return fallback;
}

inline uint64_t integer(const std::string& value, uint64_t maximum, bool allow_zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Expected nonnegative integer: " + value);
    const auto result = std::stoull(value);
    if ((!allow_zero && !result) || result > maximum) throw std::runtime_error("Integer out of range: " + value);
    return result;
}

inline hipDeviceProp_t select_device(int argc, char** argv) {
    const int device = static_cast<int>(integer(option(argc, argv, "--device", "0"), INT32_MAX, true));
    int count = 0;
    HIP_CHECK(hipGetDeviceCount(&count));
    if (device >= count) throw std::runtime_error("--device exceeds visible HIP device count");
    HIP_CHECK(hipSetDevice(device));
    hipDeviceProp_t properties{};
    HIP_CHECK(hipGetDeviceProperties(&properties, device));
    return properties;
}

inline std::string json_string(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"';
    return out.str();
}

template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), count * sizeof(T))); }
    ~Buffer() { if (data) (void)hipFree(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

template<class Launch> double elapsed_ms(Launch launch, int iterations) {
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iterations; ++i) launch();
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
    if (!(ms > 0) || !std::isfinite(ms)) throw std::runtime_error("Invalid timer result");
    return ms;
}

inline void metric(const char* name, double value, const char* unit, int blocks, int sample,
                   double ms, size_t bytes, int iterations) {
    if (!(value > 0) || !std::isfinite(value)) throw std::runtime_error("Invalid measurement");
    std::cout << std::setprecision(12) << "TILESIGHT_METRIC_JSON {\"name\":" << json_string(name)
              << ",\"value\":" << value << ",\"unit\":" << json_string(unit)
              << ",\"sample\":" << sample << ",\"time_ms\":" << ms
              << ",\"iterations\":" << iterations;
    if (blocks > 0) std::cout << ",\"blocks\":" << blocks;
    if (bytes) std::cout << ",\"bytes\":" << bytes;
    std::cout << "}\n";
}

#if defined(TILESIGHT_AMD_KERNELS)
// Avalanche each element independently. Copy payloads contain nonuniform nonzero
// 32-bit words; arithmetic/GEMM inputs map the same hash to finite positive values.
__host__ __device__ inline uint32_t payload_word(uint32_t i) {
    i ^= 0x9e3779b9u;
    i ^= i >> 16;
    i *= 0x7feb352du;
    i ^= i >> 15;
    i *= 0x846ca68bu;
    i ^= i >> 16;
    return i ? i : 1u;
}

template<class T> __host__ __device__ T numeric_value(size_t index, uint32_t seed) {
    return T(0.25) + T(payload_word(uint32_t(index) ^ seed) & 0xffffffu) / T(16777216.0);
}

template<class T> __global__ void initialize_numeric(T* data, size_t count, uint32_t seed) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        data[i] = numeric_value<T>(i, seed);
}
#endif
