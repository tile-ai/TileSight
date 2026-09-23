// Directed CUDA peer copies. No host-staged fallback is benchmarked.
// time_us is CUDA-event elapsed time per copy in the source device's stream;
// it includes submission/queue overhead and is not pure interconnect latency.
#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
#define CUDA_CHECK(operation) check((operation), #operation)

struct Options {
    std::vector<int> devices;
    size_t min_bytes = 8;
    size_t max_bytes = 67108864;
    size_t factor = 2;
    int iters = 100;
    int warmup = 10;
};

static unsigned long long integer(const std::string& value, bool allow_zero = false) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid integer: " + value);
    errno = 0;
    char* end = nullptr;
    unsigned long long number = std::strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || *end || (!allow_zero && !number))
        throw std::runtime_error("Integer out of range: " + value);
    return number;
}

static int small_integer(const std::string& value, bool allow_zero = false) {
    auto number = integer(value, allow_zero);
    if (number > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Integer exceeds INT_MAX: " + value);
    return static_cast<int>(number);
}

static size_t byte_count(const std::string& value) {
    auto number = integer(value);
    if (number > std::numeric_limits<size_t>::max())
        throw std::runtime_error("Byte count exceeds SIZE_MAX: " + value);
    return static_cast<size_t>(number);
}

static Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout << "Usage: p2p --devices 0,1 [--min-bytes 8] [--max-bytes 67108864]\n"
                         "           [--factor 2] [--iters 100] [--warmup 10]\n"
                         "Devices are ordinals inside CUDA_VISIBLE_DEVICES.\n"
                         "time_us: CUDA-event mean per peer copy, including stream\n"
                         "submission/queue overhead; not pure fabric latency.\n";
            std::exit(0);
        }
        if (++i >= argc) throw std::runtime_error("Missing value for " + key);
        std::string value = argv[i];
        if (key == "--devices") {
            if (!opt.devices.empty()) throw std::runtime_error("Specify --devices only once");
            size_t begin = 0;
            do {
                size_t end = value.find(',', begin);
                int device = small_integer(value.substr(begin, end - begin), true);
                if (std::find(opt.devices.begin(), opt.devices.end(), device) != opt.devices.end())
                    throw std::runtime_error("Duplicate device ordinal");
                opt.devices.push_back(device);
                if (end == std::string::npos) break;
                begin = end + 1;
            } while (true);
        } else if (key == "--min-bytes") opt.min_bytes = byte_count(value);
        else if (key == "--max-bytes") opt.max_bytes = byte_count(value);
        else if (key == "--factor") opt.factor = byte_count(value);
        else if (key == "--iters") opt.iters = small_integer(value);
        else if (key == "--warmup") opt.warmup = small_integer(value, true);
        else throw std::runtime_error("Unknown option: " + key);
    }
    if (opt.devices.size() < 2) throw std::runtime_error("Select at least two distinct GPUs with --devices");
    if (opt.min_bytes > opt.max_bytes) throw std::runtime_error("--min-bytes exceeds --max-bytes");
    if (opt.factor < 2) throw std::runtime_error("--factor must be at least 2");
    return opt;
}

// Peer state and allocations are local to this process. Never reset a GPU.
struct PairResources {
    int src, dst;
    void* source = nullptr;
    void* destination = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    bool enabled_forward = false, enabled_reverse = false;

    PairResources(int source_device, int destination_device)
        : src(source_device), dst(destination_device) {}
    PairResources(const PairResources&) = delete;
    PairResources& operator=(const PairResources&) = delete;

    ~PairResources() {
        cudaSetDevice(src);
        if (stream) cudaStreamSynchronize(stream);
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
        if (stream) cudaStreamDestroy(stream);
        if (source) cudaFree(source);
        if (enabled_forward) cudaDeviceDisablePeerAccess(dst);
        cudaSetDevice(dst);
        if (destination) cudaFree(destination);
        if (enabled_reverse) cudaDeviceDisablePeerAccess(src);
    }

    static bool enable(int device, int peer) {
        CUDA_CHECK(cudaSetDevice(device));
        cudaError_t result = cudaDeviceEnablePeerAccess(peer, 0);
        if (result == cudaErrorPeerAccessAlreadyEnabled) {
            // Clear only this expected error; preserve pre-existing peer state.
            cudaGetLastError();
            return false;
        }
        check(result, "cudaDeviceEnablePeerAccess");
        return true;
    }

    void initialize(size_t bytes) {
        enabled_forward = enable(src, dst);
        int reverse = 0;
        CUDA_CHECK(cudaDeviceCanAccessPeer(&reverse, dst, src));
        if (reverse) enabled_reverse = enable(dst, src);
        CUDA_CHECK(cudaSetDevice(dst));
        CUDA_CHECK(cudaMalloc(&destination, bytes));
        CUDA_CHECK(cudaSetDevice(src));
        CUDA_CHECK(cudaMalloc(&source, bytes));
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
};

static void measure_pair(const Options& opt, int src, int dst,
                         const std::vector<size_t>& sizes) {
    int can_access = 0, native_atomics = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, src, dst));
    if (!can_access) {
        for (size_t bytes : sizes)
            std::cout << src << ',' << dst << ',' << bytes << ",0,0,unsupported,,\n";
        std::cout.flush();
        return;
    }
    CUDA_CHECK(cudaDeviceGetP2PAttribute(&native_atomics, cudaDevP2PAttrNativeAtomicSupported, src, dst));

    PairResources resources(src, dst);
    resources.initialize(opt.max_bytes);
    std::vector<unsigned char> expected(opt.max_bytes), actual(opt.max_bytes);
    uint32_t state = 0x9e3779b9u ^ (uint32_t(src) + 1u);
    for (auto& value : expected) {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        value = static_cast<unsigned char>(state);
    }
    CUDA_CHECK(cudaMemcpy(resources.source, expected.data(), opt.max_bytes, cudaMemcpyHostToDevice));
    // Source initialization completes before the source-device copy stream runs.
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t bytes : sizes) {
        CUDA_CHECK(cudaSetDevice(src));
        for (int i = 0; i < opt.warmup; ++i)
            CUDA_CHECK(cudaMemcpyPeerAsync(resources.destination, dst, resources.source, src, bytes, resources.stream));
        CUDA_CHECK(cudaStreamSynchronize(resources.stream));

        // Validation must exercise the measured copies, not just warmup copies.
        CUDA_CHECK(cudaSetDevice(dst));
        CUDA_CHECK(cudaMemset(resources.destination, 0, bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaSetDevice(src));
        CUDA_CHECK(cudaEventRecord(resources.start, resources.stream));
        for (int i = 0; i < opt.iters; ++i)
            CUDA_CHECK(cudaMemcpyPeerAsync(resources.destination, dst, resources.source, src, bytes, resources.stream));
        CUDA_CHECK(cudaEventRecord(resources.stop, resources.stream));
        CUDA_CHECK(cudaEventSynchronize(resources.stop));
        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, resources.start, resources.stop));
        double time_us = double(elapsed_ms) * 1000.0 / opt.iters;
        double bandwidth = double(bytes) / time_us / 1000.0;
        if (!std::isfinite(time_us) || time_us <= 0 || !std::isfinite(bandwidth) || bandwidth <= 0)
            throw std::runtime_error("Invalid CUDA-event measurement");

        CUDA_CHECK(cudaSetDevice(dst));
        CUDA_CHECK(cudaMemcpy(actual.data(), resources.destination, bytes, cudaMemcpyDeviceToHost));
        auto mismatch = std::mismatch(expected.begin(), expected.begin() + bytes, actual.begin());
        if (mismatch.first != expected.begin() + bytes)
            throw std::runtime_error("P2P validation failed for " + std::to_string(src) + " -> " +
                                     std::to_string(dst) + " at byte " +
                                     std::to_string(mismatch.first - expected.begin()));
        std::cout << src << ',' << dst << ',' << bytes << ",1," << native_atomics
                  << ",measured," << std::setprecision(12) << time_us << ',' << bandwidth << '\n';
        std::cout.flush();
    }
}

int main(int argc, char** argv) {
    try {
        Options opt = parse_options(argc, argv);
        int count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&count));
        for (int device : opt.devices) {
            if (device >= count) throw std::runtime_error("Device ordinal is outside the current CUDA visibility mask");
            cudaDeviceProp prop{};
            CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            std::cerr << "Visible device " << device << ": " << prop.name << '\n';
        }
        std::vector<size_t> sizes;
        for (size_t bytes = opt.min_bytes;;) {
            sizes.push_back(bytes);
            if (bytes == opt.max_bytes) break;
            bytes = bytes > opt.max_bytes / opt.factor ? opt.max_bytes : bytes * opt.factor;
        }
        std::cerr << "time_us is CUDA-event mean per copy in the source-device stream, including\n"
                     "submission/queue overhead; it is not pure fabric latency. Host-staged fallback is skipped.\n";
        std::cout << "src,dst,bytes,can_access_peer,peer_native_atomics,status,time_us,bandwidth_gb_s\n";
        for (int src : opt.devices)
            for (int dst : opt.devices)
                if (src != dst) measure_pair(opt, src, dst, sizes);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
