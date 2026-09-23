#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "inline_ptx_func.hpp"

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
  int blocks_per_sm = 2;
  int threads = 128;
  int iters = 20000;
  int warmup = 2;
  int repeat = 10;
  std::string data = "nonzero";
};

constexpr int kMmaM = 128;
constexpr int kMmaN = 128;
constexpr int kMmaK = 16;

static int parse_positive(const char* s) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0' || v <= 0) {
    std::cerr << "Invalid positive integer: " << s << "\n";
    std::exit(1);
  }
  return int(v);
}

static int parse_nonnegative(const char* s) {
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0' || v < 0) {
    std::cerr << "Invalid nonnegative integer: " << s << "\n";
    std::exit(1);
  }
  return int(v);
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
      opt.device = parse_nonnegative(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--blocks-per-sm")) {
      opt.blocks_per_sm = parse_positive(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--threads")) {
      opt.threads = parse_positive(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--iters")) {
      opt.iters = parse_positive(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      opt.warmup = parse_nonnegative(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--repeat")) {
      opt.repeat = parse_positive(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--data")) {
      opt.data = need_arg(argv[i]);
      if (opt.data != "zero" && opt.data != "nonzero" &&
          opt.data != "alt" && opt.data != "random") {
        std::cerr << "Unsupported data mode: " << opt.data << "\n";
        std::exit(1);
      }
    } else {
      std::cerr << "Unknown option: " << argv[i] << "\n";
      std::exit(1);
    }
  }
  return opt;
}

__device__ uint32_t make_synthetic_word(int data_mode, int idx) {
  if (data_mode == 0) return 0u;
  if (data_mode == 1) return 0x3c003c00u;  // two fp16 1.0 values
  if (data_mode == 2) {
    constexpr uint32_t vals[] = {
        0x3c003c00u,  // 1.0, 1.0
        0xbc003c00u,  // -1.0, 1.0
        0x3800b800u,  // 0.5, -0.5
        0x40003400u,  // 2.0, 0.25
    };
    return vals[idx & 3];
  }
  uint32_t x = uint32_t(idx) * 747796405u + 2891336453u;
  x = ((x >> ((x >> 28) + 4)) ^ x) * 277803737u;
  x = (x >> 22) ^ x;
  // Keep the optional random issue pattern finite and nonzero in both FP16
  // lanes. Arbitrary integer bit patterns could introduce NaNs and infinities.
  uint32_t low = 0x3800u | (x & 0x3ffu);
  uint32_t high = 0x3800u | ((x >> 16) & 0x3ffu);
  return low | (high << 16);
}

__global__ void tcgen05_mma_perf_kernel(int iters, int data_mode,
                                        unsigned long long* sink) {
  constexpr int kSharedWords = 8192 + 256;
  // The shared-memory descriptor uses this as a leading-dimension layout
  // parameter. The tcgen05 f16 MMA instruction's actual K tile is 16.
  constexpr int kSmemK = 64;
  constexpr int kTmemColumns = 512;
  constexpr int kProbeRep = 8;
  __shared__ __align__(1024) uint32_t sharedMem[kSharedWords];
  __shared__ uint32_t allocated_tmem;
  __shared__ uint64_t mma_done;

  int tid = threadIdx.x;
  int warp_id = tid / 32;

  for (int i = tid; i < kSharedWords; i += blockDim.x) {
    sharedMem[i] = make_synthetic_word(data_mode, i);
  }
  const uint32_t barrier_addr = cast_smem_ptr_to_uint(&mma_done);
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"
                 :: "r"(barrier_addr) : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // Publish generic shared-memory stores to the tensor-core async proxy.
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();

  if (warp_id == 0) {
    // The allocation result must not overwrite an element of matrix A.
    tmem_allocate(&allocated_tmem, kTmemColumns);
  }
  __syncthreads();

  uint32_t tmem_ptr = allocated_tmem;
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (tid == 0) {
    InstrDescriptor idesc = make_instr_desc<kMmaM, kMmaN>();
    SmemDescriptor desc_a = make_smem_desc<kMmaM, kSmemK>(
        reinterpret_cast<uint16_t*>(sharedMem));
    SmemDescriptor desc_b = make_smem_desc<kMmaM, kSmemK>(
        reinterpret_cast<uint16_t*>(sharedMem + 4096));
    for (int i = 0; i < iters; ++i) {
      amma_fp16bf16_ss<kMmaM, kMmaN>(uint64_t(desc_a), uint64_t(desc_b),
                                     tmem_ptr, uint32_t(idesc), i != 0);
    }
    // A CTA barrier alone cannot wait for asynchronous MMA completion.
    amma_commit(&mma_done);
  }
  asm volatile(
      "{\n\t"
      ".reg .pred complete;\n\t"
      "wait_for_mma:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 complete, [%0], 0;\n\t"
      "@!complete bra wait_for_mma;\n\t"
      "}\n"
      :: "r"(barrier_addr) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  uint32_t probe[kProbeRep];
  // Each warp can access only its own 32-row quarter of TMEM.
  const uint32_t probe_ptr = tmem_ptr + ((warp_id % 4) * 32u << 16);
  tmem_ld_32dp32bNx<kProbeRep>(probe_ptr, probe);
  fence_view_async_tmem_load();

  unsigned long long local = 0;
  #pragma unroll
  for (int i = 0; i < kProbeRep; ++i) {
    local += probe[i];
  }
  if (tid == 0) {
    sink[blockIdx.x] = local;
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

  if (warp_id == 0) {
    tmem_free(tmem_ptr, kTmemColumns);
  }
  if (tid == 0) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];"
                 :: "r"(barrier_addr) : "memory");
  }
}

int main(int argc, char** argv) {
  Options opt = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  if (prop.major != 10 || prop.minor != 0) {
    std::cerr << "This tcgen05 benchmark requires SM100 (B100/B200/GB200).\n";
    return 1;
  }
  int blocks = prop.multiProcessorCount * opt.blocks_per_sm;
  int data_mode = opt.data == "zero" ? 0 : (opt.data == "nonzero" ? 1 : (opt.data == "alt" ? 2 : 3));

  unsigned long long* d_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sink, sizeof(unsigned long long) * blocks));

  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto launch_once = [&]() {
    tcgen05_mma_perf_kernel<<<blocks, opt.threads>>>(opt.iters, data_mode,
                                                     d_sink);
    CUDA_CHECK(cudaGetLastError());
  };

  for (int i = 0; i < opt.warmup; ++i) {
    launch_once();
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> ms;
  ms.reserve(opt.repeat);
  for (int i = 0; i < opt.repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start));
    launch_once();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    ms.push_back(elapsed);
  }

  std::sort(ms.begin(), ms.end());
  double sum_ms = 0.0;
  for (float t : ms) sum_ms += t;
  double min_ms = ms.front();
  double p50_ms = ms[ms.size() / 2];
  double avg_ms = sum_ms / double(ms.size());

  constexpr double flop_per_mma =
      2.0 * double(kMmaM) * double(kMmaN) * double(kMmaK);
  double total_flop = double(blocks) * double(opt.iters) * flop_per_mma;
  auto tflops = [&](double elapsed_ms) {
    return total_flop / (elapsed_ms * 1.0e-3) / 1.0e12;
  };

  std::cout << "Device " << opt.device << ": " << prop.name << ", SM "
            << prop.major << "." << prop.minor << ", SMs "
            << prop.multiProcessorCount << "\n";
  std::cout << "Kernel: inline tcgen05.mma.cta_group::1.kind::f16"
            << ", synthetic shared-memory data=" << opt.data
            << ", blocks_per_sm=" << opt.blocks_per_sm
            << ", threads=" << opt.threads << ", iters=" << opt.iters << "\n";
  std::cout << std::fixed << std::setprecision(3)
            << "min_ms " << min_ms << ", p50_ms " << p50_ms
            << ", avg_ms " << avg_ms << "\n"
            << "min/p50/avg TFLOPS: " << tflops(min_ms) << " / "
            << tflops(p50_ms) << " / " << tflops(avg_ms) << "\n";

  std::cout << "TILESIGHT_METRIC tcgen05_fp16 " << tflops(p50_ms) << " TFLOP/s\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_sink));
  return 0;
}
