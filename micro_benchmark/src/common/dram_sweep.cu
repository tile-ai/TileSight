#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <climits>

#include <cuda_runtime.h>
#include "data_pattern.hpp"

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err__ = (call);                                               \
    if (err__ != cudaSuccess) {                                               \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,     \
                   cudaGetErrorString(err__));                               \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

static constexpr int kDefaultThreads = 256;
static constexpr int kDefaultIters = 4096;
static constexpr int kDefaultUnroll = 8;
static constexpr size_t kDefaultBytes = size_t{8} << 30;

struct Args {
  int device = 0;
  int threads = kDefaultThreads;
  int iters = kDefaultIters;
  int unroll = kDefaultUnroll;
  int repeats = 3;
  size_t bytes = kDefaultBytes;
  size_t guard_kb = 128;
  uint32_t seed = tilesight_bench::kDefaultDataSeed;
  std::vector<int> blocks;
  std::vector<std::string> methods{"read_cg", "cp_async", "write_cs", "copy_cg_cs"};
};

static std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    if (!item.empty()) out.push_back(item);
  }
  return out;
}

static std::vector<int> split_ints(const std::string &s) {
  std::vector<int> out;
  for (auto &part : split(s, ',')) out.push_back(std::stoi(part));
  return out;
}

static void usage(const char *argv0) {
  std::fprintf(stderr,
      "Usage: %s [--device N] [--blocks 1,2,4] [--methods read_cg,cp_async,write_cs,copy_cg_cs]\n"
      "          [--threads 256] [--iters 4096] [--unroll 8] [--bytes BYTES]\n"
      "          [--guard-kb 128] [--repeats 3] [--seed 1729]\n",
      argv0);
}

static Args parse_args(int argc, char **argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "Missing value for %s\n", name);
        usage(argv[0]);
        std::exit(2);
      }
      return argv[++i];
    };
    if (key == "--device") args.device = std::stoi(need_value("--device"));
    else if (key == "--threads") args.threads = std::stoi(need_value("--threads"));
    else if (key == "--iters") args.iters = std::stoi(need_value("--iters"));
    else if (key == "--unroll") args.unroll = std::stoi(need_value("--unroll"));
    else if (key == "--repeats") args.repeats = std::stoi(need_value("--repeats"));
    else if (key == "--bytes") args.bytes = std::stoull(need_value("--bytes"));
    else if (key == "--guard-kb") args.guard_kb = std::stoull(need_value("--guard-kb"));
    else if (key == "--blocks") args.blocks = split_ints(need_value("--blocks"));
    else if (key == "--methods") args.methods = split(need_value("--methods"), ',');
    else if (key == "--seed") {
      if (!tilesight_bench::parse_data_seed(need_value("--seed"), &args.seed)) {
        std::fprintf(stderr, "--seed must be an unsigned 32-bit integer\n");
        std::exit(2);
      }
    }
    else if (key == "--help" || key == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "Unknown argument: %s\n", key.c_str());
      usage(argv[0]);
      std::exit(2);
    }
  }
  return args;
}

__host__ __device__ static uint4 payload(size_t index, uint32_t seed) {
  return make_uint4(tilesight_bench::data_word(4 * uint64_t(index), seed),
                    tilesight_bench::data_word(4 * uint64_t(index) + 1, seed),
                    tilesight_bench::data_word(4 * uint64_t(index) + 2, seed),
                    tilesight_bench::data_word(4 * uint64_t(index) + 3, seed));
}

__global__ void initialize_payload(uint4* data, size_t count, uint32_t seed) {
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += size_t(blockDim.x) * gridDim.x)
    data[i] = payload(i, seed);
}

__device__ __forceinline__ uint4 ldg_cg(const void *ptr) {
  uint4 ret;
  asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
               : "=r"(ret.x), "=r"(ret.y), "=r"(ret.z), "=r"(ret.w)
               : "l"(ptr) : "memory");
  return ret;
}

__device__ __forceinline__ void stg_cs(const uint4 &reg, void *ptr) {
  asm volatile("st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
               :
               : "r"(reg.x), "r"(reg.y), "r"(reg.z), "r"(reg.w), "l"(ptr) : "memory");
}

__device__ __forceinline__ unsigned smem_addr(void *ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg(void *smem_ptr, const void *gmem_ptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(smem_addr(smem_ptr)), "l"(gmem_ptr)
               : "memory");
}

template <int UNROLL>
__global__ void read_cg_kernel(const uint4 *__restrict__ src,
                               uint4 *__restrict__ sink,
                               size_t elem_mask,
                               int iters) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  size_t tid = threadIdx.x;
  size_t block_base = size_t(blockIdx.x) * blockDim.x * UNROLL;
  size_t grid_step = size_t(gridDim.x) * blockDim.x * UNROLL;

  for (int iter = 0; iter < iters; ++iter) {
    size_t base = size_t(iter) * grid_step + block_base + tid;
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      uint4 v = ldg_cg(src + ((base + size_t(j) * blockDim.x) & elem_mask));
      acc.x ^= v.x;
      acc.y += v.y;
      acc.z ^= v.z;
      acc.w += v.w;
    }
  }
  sink[size_t(blockIdx.x) * blockDim.x + tid] = acc;
}

template <int UNROLL>
__global__ void read_ca_kernel(const uint4 *__restrict__ src,
                               uint4 *__restrict__ sink,
                               size_t elem_mask,
                               int iters) {
  uint4 acc = make_uint4(0, 0, 0, 0);
  size_t tid = threadIdx.x;
  size_t block_base = size_t(blockIdx.x) * blockDim.x * UNROLL;
  size_t grid_step = size_t(gridDim.x) * blockDim.x * UNROLL;

  for (int iter = 0; iter < iters; ++iter) {
    size_t base = size_t(iter) * grid_step + block_base + tid;
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      uint4 v = src[(base + size_t(j) * blockDim.x) & elem_mask];
      acc.x ^= v.x;
      acc.y += v.y;
      acc.z ^= v.z;
      acc.w += v.w;
    }
  }
  sink[size_t(blockIdx.x) * blockDim.x + tid] = acc;
}

template <int UNROLL>
__global__ void write_cs_kernel(uint4 *__restrict__ dst,
                                size_t elem_mask,
                                int iters,
                                uint32_t seed) {
  // Precompute register operands once, outside the store loop. The pattern
  // repeats across blocks but varies across every word of a cache line.
  // Its period divides both blockDim and the allocation, so wraparound
  // writers always store identical values at any shared destination address.
  const size_t thread_mask = size_t(blockDim.x & -blockDim.x) - 1;
  const size_t pattern_mask = thread_mask < elem_mask ? thread_mask : elem_mask;
  uint4 value = payload(threadIdx.x & pattern_mask, seed);
  size_t tid = threadIdx.x;
  size_t block_base = size_t(blockIdx.x) * blockDim.x * UNROLL;
  size_t grid_step = size_t(gridDim.x) * blockDim.x * UNROLL;

  for (int iter = 0; iter < iters; ++iter) {
    size_t base = size_t(iter) * grid_step + block_base + tid;
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      stg_cs(value, dst + ((base + size_t(j) * blockDim.x) & elem_mask));
    }
  }
}

template <int UNROLL>
__global__ void copy_cg_cs_kernel(const uint4 *__restrict__ src,
                                  uint4 *__restrict__ dst,
                                  size_t elem_mask,
                                  int iters) {
  size_t tid = threadIdx.x;
  size_t block_base = size_t(blockIdx.x) * blockDim.x * UNROLL;
  size_t grid_step = size_t(gridDim.x) * blockDim.x * UNROLL;

  for (int iter = 0; iter < iters; ++iter) {
    size_t base = size_t(iter) * grid_step + block_base + tid;
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      size_t idx = (base + size_t(j) * blockDim.x) & elem_mask;
      uint4 v = ldg_cg(src + idx);
      stg_cs(v, dst + idx);
    }
  }
}

template <int UNROLL>
__global__ void cp_async_kernel(const uint4 *__restrict__ src,
                                uint4 *__restrict__ sink,
                                size_t elem_mask,
                                int iters,
                                size_t guard_bytes) {
  extern __shared__ unsigned char raw_smem[];
  size_t aligned_guard = (guard_bytes + 15u) & ~size_t(15u);
  uint4 *tile = reinterpret_cast<uint4 *>(raw_smem + aligned_guard);

  uint4 acc = make_uint4(0, 0, 0, 0);
  size_t tid = threadIdx.x;
  size_t block_base = size_t(blockIdx.x) * blockDim.x * UNROLL;
  size_t grid_step = size_t(gridDim.x) * blockDim.x * UNROLL;

  for (int iter = 0; iter < iters; ++iter) {
    size_t base = size_t(iter) * grid_step + block_base + tid;
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      cp_async_cg(tile + j * blockDim.x + tid,
                  src + ((base + size_t(j) * blockDim.x) & elem_mask));
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
    __syncthreads();
#pragma unroll
    for (int j = 0; j < UNROLL; ++j) {
      uint4 v = tile[j * blockDim.x + tid];
      acc.x ^= v.x;
      acc.y += v.y;
      acc.z ^= v.z;
      acc.w += v.w;
    }
    __syncthreads();
  }
  sink[size_t(blockIdx.x) * blockDim.x + tid] = acc;
}

using KernelPtr = void (*)();

template <int UNROLL>
static void *kernel_for_method(const std::string &method) {
  if (method == "read_cg") return reinterpret_cast<void *>(read_cg_kernel<UNROLL>);
  if (method == "read_ca") return reinterpret_cast<void *>(read_ca_kernel<UNROLL>);
  if (method == "write_cs") return reinterpret_cast<void *>(write_cs_kernel<UNROLL>);
  if (method == "copy_cg_cs") return reinterpret_cast<void *>(copy_cg_cs_kernel<UNROLL>);
  if (method == "cp_async") return reinterpret_cast<void *>(cp_async_kernel<UNROLL>);
  return nullptr;
}

template <int UNROLL>
static int occupancy_for_method(const std::string &method, int threads, size_t smem_bytes) {
  int active = 0;
  if (method == "read_cg") {
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, read_cg_kernel<UNROLL>, threads, smem_bytes));
  } else if (method == "read_ca") {
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, read_ca_kernel<UNROLL>, threads, smem_bytes));
  } else if (method == "write_cs") {
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, write_cs_kernel<UNROLL>, threads, smem_bytes));
  } else if (method == "copy_cg_cs") {
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, copy_cg_cs_kernel<UNROLL>, threads, smem_bytes));
  } else if (method == "cp_async") {
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, cp_async_kernel<UNROLL>, threads, smem_bytes));
  }
  return active;
}

template <int UNROLL>
static void set_smem_attr(const std::string &method, size_t smem_bytes) {
  void *kernel = kernel_for_method<UNROLL>(method);
  if (!kernel) return;
  CHECK_CUDA(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  static_cast<int>(smem_bytes)));
}

template <int UNROLL>
static float launch_once(const std::string &method,
                         int blocks,
                         int threads,
                         int iters,
                         size_t elem_mask,
                         size_t guard_bytes,
                         size_t smem_bytes,
                         const uint4 *src,
                         uint4 *dst,
                         uint32_t seed) {
  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  if (method == "read_cg") {
    read_cg_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters);
  } else if (method == "read_ca") {
    read_ca_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters);
  } else if (method == "write_cs") {
    write_cs_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(dst, elem_mask, iters, seed);
  } else if (method == "copy_cg_cs") {
    copy_cg_cs_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters);
  } else if (method == "cp_async") {
    cp_async_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters, guard_bytes);
  } else {
    std::fprintf(stderr, "Unknown method: %s\n", method.c_str());
    std::exit(2);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventRecord(start));
  if (method == "read_cg") {
    read_cg_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters);
  } else if (method == "read_ca") {
    read_ca_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters);
  } else if (method == "write_cs") {
    write_cs_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(dst, elem_mask, iters, seed);
  } else if (method == "copy_cg_cs") {
    copy_cg_cs_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters);
  } else if (method == "cp_async") {
    cp_async_kernel<UNROLL><<<blocks, threads, smem_bytes>>>(src, dst, elem_mask, iters, guard_bytes);
  }
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));
  return ms;
}

static bool is_power_of_two(size_t x) {
  return x && ((x & (x - 1)) == 0);
}

template <int UNROLL>
static void validate_samples(const std::string& method, const uint4* data,
                             size_t elems, int blocks, int threads, int iters,
                             uint32_t seed) {
  const bool stores = method == "write_cs" || method == "copy_cg_cs";
  const size_t workers = size_t(blocks) * threads;
  const size_t count = stores ? elems : workers;
  for (size_t sample : {size_t(0), count / 2, count - 1}) {
    uint4 actual, expected = make_uint4(0, 0, 0, 0);
    CHECK_CUDA(cudaMemcpy(&actual, data + sample, sizeof(actual), cudaMemcpyDeviceToHost));
    if (method == "copy_cg_cs") expected = payload(sample, seed);
    else if (method == "write_cs") {
      const size_t pattern_mask = std::min(size_t(threads & -threads) - 1, elems - 1);
      expected = payload(sample & pattern_mask, seed);
    } else {
      const size_t tid = sample % threads;
      const size_t block_base = (sample / threads) * threads * UNROLL;
      const size_t step = workers * UNROLL;
      for (int iter = 0; iter < iters; ++iter)
        for (int j = 0; j < UNROLL; ++j) {
          uint4 value = payload((size_t(iter) * step + block_base + tid + size_t(j) * threads) & (elems - 1), seed);
          expected.x ^= value.x; expected.y += value.y;
          expected.z ^= value.z; expected.w += value.w;
        }
    }
    if (actual.x != expected.x || actual.y != expected.y || actual.z != expected.z || actual.w != expected.w) {
      std::fprintf(stderr, "Payload validation failed: method=%s sample=%zu\n", method.c_str(), sample);
      std::exit(1);
    }
  }
}

template <int UNROLL>
static int run(const Args &args) {
  if (args.threads <= 0 || args.threads > 1024 || args.threads % 32 != 0 ||
      args.iters <= 0 || args.repeats <= 0 || args.methods.empty()) {
    std::fprintf(stderr, "Use positive iterations/repeats and 32..1024 threads in full warps\n");
    return 2;
  }
  CHECK_CUDA(cudaSetDevice(args.device));

  cudaDeviceProp prop{};
  CHECK_CUDA(cudaGetDeviceProperties(&prop, args.device));
  int sm_count = prop.multiProcessorCount;

  std::vector<int> blocks = args.blocks;
  if (blocks.empty()) {
    int defaults[] = {1, 2, 4, 8, 16, 32, 64, sm_count, sm_count * 2, sm_count * 4, sm_count * 8};
    blocks.assign(defaults, defaults + sizeof(defaults) / sizeof(defaults[0]));
    std::sort(blocks.begin(), blocks.end());
    blocks.erase(std::unique(blocks.begin(), blocks.end()), blocks.end());
  }

  size_t bytes = args.bytes;
  size_t elems = bytes / sizeof(uint4);
  if (bytes % sizeof(uint4) != 0 || !is_power_of_two(elems)) {
    std::fprintf(stderr, "--bytes / sizeof(uint4) must be a power of two for mask indexing\n");
    return 2;
  }
  size_t elem_mask = elems - 1;
  for (int value : blocks) {
    if (value <= 0 || size_t(value) * args.threads > elems) {
      std::fprintf(stderr, "Each block count must be positive and its thread count fit the allocation\n");
      return 2;
    }
  }

  uint4 *src = nullptr;
  uint4 *dst = nullptr;
  CHECK_CUDA(cudaMalloc(&src, bytes));
  CHECK_CUDA(cudaMalloc(&dst, bytes));
  initialize_payload<<<sm_count * 4, 256>>>(src, elems, args.seed);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemset(dst, 0, bytes));

  std::printf("method,device,gpu_name,sm_count,blocks,threads,blocks_per_sm,resident_blocks_per_sm,guard_kb,iters,unroll,repeat,time_ms,moved_GiB,achieved_ddr_bw_GiB_s,achieved_ddr_bw_GB_s,data_pattern,data_seed,validation\n");

  for (const auto &method : args.methods) {
    if (!kernel_for_method<UNROLL>(method)) {
      std::fprintf(stderr, "Unknown method: %s\n", method.c_str());
      return 2;
    }

    size_t guard_bytes = args.guard_kb * 1024;
    size_t smem_bytes = guard_bytes;
    if (method == "cp_async") {
      size_t cp_bytes = size_t(args.threads) * UNROLL * sizeof(uint4);
      smem_bytes = ((guard_bytes + 15u) & ~size_t(15u)) + cp_bytes;
    }
    set_smem_attr<UNROLL>(method, smem_bytes);
    int resident = occupancy_for_method<UNROLL>(method, args.threads, smem_bytes);

    for (int blocks_i : blocks) {
      if (blocks_i <= 0) continue;
      // A small grid must still traverse the full out-of-cache allocation.
      const size_t elems_per_iter = size_t(blocks_i) * args.threads * UNROLL;
      const size_t required_iters = std::max(size_t(args.iters), (elems + elems_per_iter - 1) / elems_per_iter);
      if (required_iters > INT_MAX) {
        std::fprintf(stderr, "Requested allocation requires too many iterations\n");
        return 2;
      }
      const int timed_iters = int(required_iters);
      for (int r = 0; r < args.repeats; ++r) {
        float ms = launch_once<UNROLL>(method, blocks_i, args.threads, timed_iters,
                                       elem_mask, guard_bytes, smem_bytes, src, dst, args.seed);
        if (!std::isfinite(ms) || ms <= 0) {
          std::fprintf(stderr, "Invalid CUDA-event duration\n");
          return 1;
        }
        validate_samples<UNROLL>(method, dst, elems, blocks_i, args.threads, timed_iters, args.seed);
        double bytes_per_elem = sizeof(uint4);
        double traffic_multiplier = (method == "copy_cg_cs") ? 2.0 : 1.0;
        double moved_bytes = double(blocks_i) * args.threads * timed_iters * UNROLL *
                             bytes_per_elem * traffic_multiplier;
        double moved_gib = moved_bytes / double(1ull << 30);
        double bw_gib_s = moved_gib / (double(ms) / 1000.0);
        double bw_gb_s = (moved_bytes / 1.0e9) / (double(ms) / 1000.0);
        std::printf("%s,%d,%s,%d,%d,%d,%.6f,%d,%zu,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%s,%u,sampled_passed\n",
                    method.c_str(), args.device, prop.name, sm_count, blocks_i, args.threads,
                    double(blocks_i) / double(sm_count), resident, args.guard_kb,
                    timed_iters, UNROLL, r, ms, moved_gib, bw_gib_s, bw_gb_s,
                    method == "write_cs" ? "thread_register_hash_v1" : "index_hash_v1", args.seed);
        std::fflush(stdout);
      }
    }
  }

  CHECK_CUDA(cudaFree(src));
  CHECK_CUDA(cudaFree(dst));
  return 0;
}

int main(int argc, char **argv) {
  Args args = parse_args(argc, argv);
  if (args.unroll == 1) return run<1>(args);
  if (args.unroll == 2) return run<2>(args);
  if (args.unroll == 4) return run<4>(args);
  if (args.unroll == 8) return run<8>(args);
  if (args.unroll == 16) return run<16>(args);
  std::fprintf(stderr, "Supported --unroll values: 1,2,4,8,16\n");
  return 2;
}
