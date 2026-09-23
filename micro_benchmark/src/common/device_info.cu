#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

static void check(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << cudaGetErrorString(error) << '\n';
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    if (argc != 3 || std::strcmp(argv[1], "--device")) return 2;
    char* end = nullptr;
    long device = std::strtol(argv[2], &end, 10);
    int count = 0;
    check(cudaGetDeviceCount(&count));
    if (!*argv[2] || *end || device < 0 || device >= count) {
        std::cerr << "Device index is outside the visible CUDA devices\n";
        return 2;
    }
    check(cudaSetDevice(int(device)));
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, int(device)));
    int driver = 0, runtime = 0;
    check(cudaDriverGetVersion(&driver));
    check(cudaRuntimeGetVersion(&runtime));
    size_t free_bytes = 0, total_bytes = 0;
    check(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::string escaped;
    for (char c : std::string(prop.name)) {
        if (c == '"' || c == '\\') escaped += '\\';
        escaped += c;
    }
    std::cout << "{\"name\":\"" << escaped << "\",\"uuid\":\"GPU-";
    for (int i = 0; i < 16; ++i) {
        if (i == 4 || i == 6 || i == 8 || i == 10) std::printf("-");
        std::printf("%02x", static_cast<unsigned char>(prop.uuid.bytes[i]));
    }
    std::cout << "\",\"visible_device\":" << device
              << ",\"compute_capability\":\"" << prop.major << '.' << prop.minor
              << "\",\"sm_count\":" << prop.multiProcessorCount
              << ",\"l2_bytes\":" << prop.l2CacheSize
              << ",\"total_memory_bytes\":" << total_bytes
              << ",\"free_memory_bytes\":" << free_bytes
              << ",\"cuda_driver_version\":" << driver
              << ",\"cuda_runtime_version\":" << runtime << "}\n";
}
