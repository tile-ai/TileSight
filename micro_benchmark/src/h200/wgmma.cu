#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t err = (stmt);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << " at "        \
                << __FILE__ << ":" << __LINE__ << "\n";                      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Options {
  int device = 0;
  int blocks_per_sm = 4;
  int iters = 20000;
  int warmup = 3;
  int repeat = 20;
  int data_pattern = 1;  // 0: zero, 1: fixed nonzero, 2: pseudo-random nonzero
  std::string data_name = "nonzero";
};

static void usage(const char* argv0) {
  std::cout << "Usage: " << argv0 << " [options]\n"
            << "  --device ID          CUDA device, default 0\n"
            << "  --blocks-per-sm N    128-thread warpgroups per SM, default 4\n"
            << "  --iters N            WGMMA instructions per block, default 20000\n"
            << "  --warmup N           Warmup kernel launches, default 3\n"
            << "  --repeat N           Timed kernel launches, default 20\n"
            << "  --data zero|nonzero|random\n";
}

static int64_t parse_i64(const char* s, bool allow_zero) {
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (!end || *end != '\0' || v < (allow_zero ? 0 : 1)) {
    std::cerr << "Invalid integer: " << s << "\n";
    std::exit(1);
  }
  return int64_t(v);
}

static Options parse_options(int argc, char** argv) {
  Options opt;
  for (int i = 1; i < argc; ++i) {
    auto need_arg = [&](const char* name) {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(1);
      }
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--device")) {
      opt.device = int(parse_i64(need_arg(argv[i]), true));
    } else if (!std::strcmp(argv[i], "--blocks-per-sm")) {
      opt.blocks_per_sm = int(parse_i64(need_arg(argv[i]), false));
    } else if (!std::strcmp(argv[i], "--iters")) {
      opt.iters = int(parse_i64(need_arg(argv[i]), false));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      opt.warmup = int(parse_i64(need_arg(argv[i]), true));
    } else if (!std::strcmp(argv[i], "--repeat")) {
      opt.repeat = int(parse_i64(need_arg(argv[i]), false));
    } else if (!std::strcmp(argv[i], "--data")) {
      opt.data_name = need_arg(argv[i]);
      if (opt.data_name == "zero") {
        opt.data_pattern = 0;
      } else if (opt.data_name == "nonzero") {
        opt.data_pattern = 1;
      } else if (opt.data_name == "random") {
        opt.data_pattern = 2;
      } else {
        std::cerr << "Unsupported --data value: " << opt.data_name << "\n";
        usage(argv[0]);
        std::exit(1);
      }
    } else if (!std::strcmp(argv[i], "--help") || !std::strcmp(argv[i], "-h")) {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "Unknown option: " << argv[i] << "\n";
      usage(argv[0]);
      std::exit(1);
    }
  }
  return opt;
}

__device__ __forceinline__ uint16_t pick_half_bits(uint32_t x) {
  switch (x & 7u) {
    case 0: return 0x3800;  // 0.5
    case 1: return 0x3a00;  // 0.75
    case 2: return 0x3c00;  // 1.0
    case 3: return 0x3d00;  // 1.25
    case 4: return 0x3e00;  // 1.5
    case 5: return 0x4000;  // 2.0
    case 6: return 0x4200;  // 3.0
    default: return 0x4400; // 4.0
  }
}

__device__ __forceinline__ uint32_t pack_half2(uint16_t lo, uint16_t hi) {
  return uint32_t(lo) | (uint32_t(hi) << 16);
}

__global__ __launch_bounds__(128, 1)
void wgmma_m64n64k16_f32_kernel(int iters, uint64_t* cycles_out,
                                float* sink_out, int data_pattern) {
  if (threadIdx.x >= 128) return;

  extern __shared__ uint16_t smem_u16[];
  for (int i = threadIdx.x; i < 16 * 64; i += blockDim.x) {
    uint16_t v = 0x3c00;
    if (data_pattern == 0) {
      v = 0;
    } else if (data_pattern == 2) {
      uint32_t h = uint32_t(i) * 1103515245u + uint32_t(blockIdx.x) * 12345u +
                   uint32_t(threadIdx.x) * 2654435761u;
      v = pick_half_bits(h >> 16);
    }
    smem_u16[i] = v;
  }
  // Publish generic shared-memory stores to WGMMA's async proxy.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  uint64_t baddr = 0;
  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(baddr) : "l"(smem_u16));
  // B is encoded as a Major-K GMMA descriptor over a logical (N,K) tile.
  // For m64n64k16 with half elements: K is two uint128_t columns, and each
  // group of 8 N rows advances 8 uint128_t units.
  uint64_t bdesc = ((baddr >> 4) & 0x3fffull) |
                   (uint64_t(64) << 16) |
                   (uint64_t(8) << 32);

  uint32_t a0 = 0x3c003c00u;
  uint32_t a1 = 0x3c003c00u;
  uint32_t a2 = 0x3c003c00u;
  uint32_t a3 = 0x3c003c00u;
  if (data_pattern == 0) {
    a0 = a1 = a2 = a3 = 0;
  } else if (data_pattern == 2) {
    uint32_t seed = uint32_t(blockIdx.x) * 747796405u +
                    uint32_t(threadIdx.x) * 2891336453u + 0x9e3779b9u;
    a0 = pack_half2(pick_half_bits(seed), pick_half_bits(seed >> 3));
    a1 = pack_half2(pick_half_bits(seed >> 6), pick_half_bits(seed >> 9));
    a2 = pack_half2(pick_half_bits(seed >> 12), pick_half_bits(seed >> 15));
    a3 = pack_half2(pick_half_bits(seed >> 18), pick_half_bits(seed >> 21));
  }
  float d00 = 0.0f, d01 = 0.0f, d02 = 0.0f, d03 = 0.0f;
  float d04 = 0.0f, d05 = 0.0f, d06 = 0.0f, d07 = 0.0f;
  float d08 = 0.0f, d09 = 0.0f, d10 = 0.0f, d11 = 0.0f;
  float d12 = 0.0f, d13 = 0.0f, d14 = 0.0f, d15 = 0.0f;
  float d16 = 0.0f, d17 = 0.0f, d18 = 0.0f, d19 = 0.0f;
  float d20 = 0.0f, d21 = 0.0f, d22 = 0.0f, d23 = 0.0f;
  float d24 = 0.0f, d25 = 0.0f, d26 = 0.0f, d27 = 0.0f;
  float d28 = 0.0f, d29 = 0.0f, d30 = 0.0f, d31 = 0.0f;

  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
  uint64_t start = clock64();

#pragma unroll 1
  for (int i = 0; i < iters; ++i) {
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        " %8,  %9,  %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31},"
        "{%32, %33, %34, %35},"
        " %36,"
        " p, 1, 1, 0;\n"
        "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04),
          "+f"(d05), "+f"(d06), "+f"(d07), "+f"(d08), "+f"(d09),
          "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14),
          "+f"(d15), "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19),
          "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23), "+f"(d24),
          "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29),
          "+f"(d30), "+f"(d31)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(bdesc), "r"(1)
        : "memory");
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
    if ((i & 7) == 7) {
      asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
    }
  }

  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
  uint64_t stop = clock64();

  if (threadIdx.x == 0) {
    cycles_out[blockIdx.x] = stop - start;
    sink_out[blockIdx.x] = d00 + d07 + d13 + d19 + d25 + d31;
  }
}

int main(int argc, char** argv) {
  Options opt = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  if (prop.major != 9) {
    std::cerr << "Require SM90/Hopper. Got SM " << prop.major << "."
              << prop.minor << "\n";
    return 1;
  }

  const int blocks = prop.multiProcessorCount * opt.blocks_per_sm;
  const int threads = 128;
  const size_t shmem_bytes = 16 * 64 * sizeof(uint16_t);
  const double flop_per_wgmma = 2.0 * 64.0 * 64.0 * 16.0;
  const double flop_per_launch =
      flop_per_wgmma * double(opt.iters) * double(blocks);

  uint64_t* d_cycles = nullptr;
  float* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(uint64_t) * blocks));
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(float) * blocks));

  std::cout << "Device " << opt.device << ": " << prop.name << ", SM "
            << prop.major << "." << prop.minor << ", SMs "
            << prop.multiProcessorCount << "\n"
            << "Kernel: inline wgmma.m64n64k16.f32.f16.f16, A reg, B smem\n"
            << "Data pattern: " << opt.data_name << "\n"
            << "Launch: blocks " << blocks << " (" << opt.blocks_per_sm
            << "/SM), threads 128, iters " << opt.iters << "\n";

  for (int i = 0; i < opt.warmup; ++i) {
    wgmma_m64n64k16_f32_kernel<<<blocks, threads, shmem_bytes>>>(
        opt.iters, d_cycles, d_sink, opt.data_pattern);
  }
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  std::vector<float> ms;
  ms.reserve(opt.repeat);

  for (int i = 0; i < opt.repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    wgmma_m64n64k16_f32_kernel<<<blocks, threads, shmem_bytes>>>(
        opt.iters, d_cycles, d_sink, opt.data_pattern);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    ms.push_back(elapsed);
  }

  std::vector<uint64_t> cycles(blocks);
  CUDA_CHECK(cudaMemcpy(cycles.data(), d_cycles, sizeof(uint64_t) * blocks,
                        cudaMemcpyDeviceToHost));
  auto [min_cycle_it, max_cycle_it] =
      std::minmax_element(cycles.begin(), cycles.end());
  double avg_cycles =
      std::accumulate(cycles.begin(), cycles.end(), 0.0) / double(cycles.size());

  std::sort(ms.begin(), ms.end());
  double avg_ms = std::accumulate(ms.begin(), ms.end(), 0.0) / double(ms.size());
  auto tflops = [&](double elapsed_ms) {
    return flop_per_launch / (elapsed_ms * 1.0e-3) / 1.0e12;
  };
  const double cycles_per_wgmma = avg_cycles / double(opt.iters);

  std::cout << std::fixed << std::setprecision(3)
            << "event min/p50/avg ms: " << ms.front() << " / "
            << ms[ms.size() / 2] << " / " << avg_ms << "\n"
            << "event min/p50/avg TFLOPS: " << tflops(ms.front()) << " / "
            << tflops(ms[ms.size() / 2]) << " / " << tflops(avg_ms) << "\n"
            << "clock64 cycles/block min/max/avg: " << *min_cycle_it << " / "
            << *max_cycle_it << " / " << avg_cycles << "\n"
            << "clock64 cycles per WGMMA/block avg: " << cycles_per_wgmma
            << "\n";


  std::cout << "TILESIGHT_METRIC wgmma_fp16 " << tflops(ms[ms.size() / 2]) << " TFLOP/s\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_cycles));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
