#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
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

#define CUBLAS_CHECK(stmt)                                                     \
  do {                                                                         \
    cublasStatus_t err = (stmt);                                               \
    if (err != CUBLAS_STATUS_SUCCESS) {                                        \
      std::cerr << "cuBLAS error " << int(err) << " at " << __FILE__ << ":"  \
                << __LINE__ << "\n";                                          \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Options {
  int device = 0;
  int64_t m = 16384;
  int64_t n = 16384;
  int64_t k = 16384;
  int warmup = 10;
  int repeat = 30;
  size_t workspace_bytes = 256ull << 20;
  bool accum_f16 = false;
};

static void usage(const char* argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "  --device ID              CUDA device, default 0\n"
      << "  --m M --n N --k K         GEMM size, default 16384^3\n"
      << "  --warmup N               Warmup iterations, default 10\n"
      << "  --repeat N               Timed iterations, default 30\n"
      << "  --workspace-mib MIB      cuBLASLt workspace, default 256\n"
      << "  --accum f32|f16          Accumulator type, default f32\n";
}

static int64_t parse_i64(const char* s) {
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (!end || *end != '\0' || v <= 0) {
    std::cerr << "Invalid positive integer: " << s << "\n";
    std::exit(1);
  }
  return int64_t(v);
}

static int64_t parse_nonnegative_i64(const char* s) {
  char* end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (!end || *end != '\0' || v < 0) {
    std::cerr << "Invalid nonnegative integer: " << s << "\n";
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
      opt.device = int(parse_nonnegative_i64(need_arg(argv[i])));
    } else if (!std::strcmp(argv[i], "--m")) {
      opt.m = parse_i64(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--n")) {
      opt.n = parse_i64(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--k")) {
      opt.k = parse_i64(need_arg(argv[i]));
    } else if (!std::strcmp(argv[i], "--warmup")) {
      opt.warmup = int(parse_i64(need_arg(argv[i])));
    } else if (!std::strcmp(argv[i], "--repeat")) {
      opt.repeat = int(parse_i64(need_arg(argv[i])));
    } else if (!std::strcmp(argv[i], "--workspace-mib")) {
      opt.workspace_bytes = size_t(parse_i64(need_arg(argv[i]))) << 20;
    } else if (!std::strcmp(argv[i], "--accum")) {
      const char* value = need_arg(argv[i]);
      if (!std::strcmp(value, "f32")) {
        opt.accum_f16 = false;
      } else if (!std::strcmp(value, "f16")) {
        opt.accum_f16 = true;
      } else {
        std::cerr << "Unsupported accumulator: " << value << "\n";
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

__global__ void init_half_kernel(__half* p, int64_t n, float value) {
  int64_t tid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t stride = int64_t(blockDim.x) * gridDim.x;
  __half h = __float2half(value);
  for (int64_t i = tid; i < n; i += stride) {
    p[i] = h;
  }
}

static void init_half(__half* p, int64_t n, float value) {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  int blocks = std::min<int64_t>(int64_t(prop.multiProcessorCount) * 8,
                                 (n + 255) / 256);
  init_half_kernel<<<blocks, 256>>>(p, n, value);
  CUDA_CHECK(cudaGetLastError());
}

template <typename ScaleT>
static cublasStatus_t run_lt_once(cublasLtHandle_t lt,
                                  cublasLtMatmulDesc_t op_desc,
                                  const ScaleT* alpha,
                                  const __half* a,
                                  cublasLtMatrixLayout_t a_desc,
                                  const __half* b,
                                  cublasLtMatrixLayout_t b_desc,
                                  const ScaleT* beta,
                                  __half* c,
                                  cublasLtMatrixLayout_t c_desc,
                                  const cublasLtMatmulAlgo_t* algo,
                                  void* workspace,
                                  size_t workspace_bytes,
                                  cudaStream_t stream) {
  return cublasLtMatmul(lt, op_desc, alpha, a, a_desc, b, b_desc, beta, c,
                        c_desc, c, c_desc, algo, workspace, workspace_bytes,
                        stream);
}

int main(int argc, char** argv) {
  Options opt = parse_options(argc, argv);
  CUDA_CHECK(cudaSetDevice(opt.device));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, opt.device));
  std::cout << "Device " << opt.device << ": " << prop.name << ", SM "
            << prop.major << "." << prop.minor << ", SMs "
            << prop.multiProcessorCount << "\n";
  if (prop.major < 7) {
    std::cerr << "This benchmark requires Tensor Core support (SM70+).\n";
    return 1;
  }

  const int64_t elems_a = opt.m * opt.k;
  const int64_t elems_b = opt.k * opt.n;
  const int64_t elems_c = opt.m * opt.n;
  const size_t bytes_a = size_t(elems_a) * sizeof(__half);
  const size_t bytes_b = size_t(elems_b) * sizeof(__half);
  const size_t bytes_c = size_t(elems_c) * sizeof(__half);

  std::cout << "GEMM: C[" << opt.m << "x" << opt.n << "] = A[" << opt.m
            << "x" << opt.k << "] * B[" << opt.k << "x" << opt.n << "]\n"
            << "Accumulator: " << (opt.accum_f16 ? "f16" : "f32") << "\n"
            << "Tensor bytes: A " << (bytes_a >> 20) << " MiB, B "
            << (bytes_b >> 20) << " MiB, C " << (bytes_c >> 20)
            << " MiB, workspace " << (opt.workspace_bytes >> 20) << " MiB\n";

  __half *a = nullptr, *b = nullptr, *c = nullptr;
  CUDA_CHECK(cudaMalloc(&a, bytes_a));
  CUDA_CHECK(cudaMalloc(&b, bytes_b));
  CUDA_CHECK(cudaMalloc(&c, bytes_c));
  init_half(a, elems_a, 0.5f);
  init_half(b, elems_b, 0.25f);
  CUDA_CHECK(cudaMemset(c, 0, bytes_c));

  void* workspace = nullptr;
  if (opt.workspace_bytes > 0) {
    CUDA_CHECK(cudaMalloc(&workspace, opt.workspace_bytes));
  }

  cudaStream_t stream = nullptr;
  CUDA_CHECK(cudaStreamCreate(&stream));

  cublasLtHandle_t lt = nullptr;
  CUBLAS_CHECK(cublasLtCreate(&lt));

  cublasComputeType_t compute_type =
      opt.accum_f16 ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
  cudaDataType_t scale_type = opt.accum_f16 ? CUDA_R_16F : CUDA_R_32F;

  cublasLtMatmulDesc_t op_desc = nullptr;
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&op_desc, compute_type, scale_type));
  cublasOperation_t trans = CUBLAS_OP_N;
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)));

  cublasLtMatrixLayout_t a_desc = nullptr, b_desc = nullptr, c_desc = nullptr;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, opt.m, opt.k,
                                          opt.m));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, opt.k, opt.n,
                                          opt.k));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, opt.m, opt.n,
                                          opt.m));

  cublasLtMatmulPreference_t pref = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &opt.workspace_bytes,
      sizeof(opt.workspace_bytes)));

  constexpr int kAlgoCount = 32;
  std::vector<cublasLtMatmulHeuristicResult_t> heuristic(kAlgoCount);
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, op_desc, a_desc, b_desc, c_desc, c_desc, pref, kAlgoCount,
      heuristic.data(), &returned));
  if (returned == 0) {
    std::cerr << "cuBLASLt did not return a usable algorithm.\n";
    return 1;
  }

  const float alpha_f32 = 1.0f;
  const float beta_f32 = 0.0f;
  const __half alpha_f16 = __float2half(1.0f);
  const __half beta_f16 = __float2half(0.0f);

  auto run_algo = [&](const cublasLtMatmulAlgo_t* algo) {
    if (opt.accum_f16) {
      return run_lt_once(lt, op_desc, &alpha_f16, a, a_desc, b, b_desc,
                         &beta_f16, c, c_desc, algo, workspace,
                         opt.workspace_bytes, stream);
    }
    return run_lt_once(lt, op_desc, &alpha_f32, a, a_desc, b, b_desc,
                       &beta_f32, c, c_desc, algo, workspace,
                       opt.workspace_bytes, stream);
  };

  cudaEvent_t start = nullptr, stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  cublasLtMatmulAlgo_t best_algo{};
  int best_algo_index = -1;
  float best_algo_ms = std::numeric_limits<float>::infinity();
  for (int i = 0; i < returned; ++i) {
    if (heuristic[i].state != CUBLAS_STATUS_SUCCESS ||
        heuristic[i].workspaceSize > opt.workspace_bytes) {
      continue;
    }
    cublasStatus_t status = run_algo(&heuristic[i].algo);
    if (status != CUBLAS_STATUS_SUCCESS) continue;
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventRecord(start, stream));
    status = run_algo(&heuristic[i].algo);
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    if (status != CUBLAS_STATUS_SUCCESS) continue;
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    if (elapsed < best_algo_ms) {
      best_algo_ms = elapsed;
      best_algo = heuristic[i].algo;
      best_algo_index = i;
    }
  }
  if (best_algo_index < 0) {
    std::cerr << "No successful cuBLASLt algorithm fit the workspace.\n";
    return 1;
  }
  const cublasLtMatmulAlgo_t* algo = &best_algo;
  std::cout << "Selected cuBLASLt heuristic index " << best_algo_index
            << " from " << returned << " candidates, search_ms "
            << best_algo_ms << "\n";

  auto run_once = [&]() { return run_algo(algo); };

  for (int i = 0; i < opt.warmup; ++i) {
    CUBLAS_CHECK(run_once());
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<float> ms;
  ms.reserve(opt.repeat);
  for (int i = 0; i < opt.repeat; ++i) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    CUBLAS_CHECK(run_once());
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
    ms.push_back(elapsed);
  }

  std::sort(ms.begin(), ms.end());
  double sum_ms = 0.0;
  for (float t : ms) sum_ms += t;
  const double min_ms = ms.front();
  const double avg_ms = sum_ms / double(ms.size());
  const double p50_ms = ms[ms.size() / 2];
  const double flop = 2.0 * double(opt.m) * double(opt.n) * double(opt.k);
  auto tflops = [&](double elapsed_ms) {
    return flop / (elapsed_ms * 1.0e-3) / 1.0e12;
  };

  std::cout << std::fixed << std::setprecision(3)
            << "min_ms " << min_ms << ", p50_ms " << p50_ms << ", avg_ms "
            << avg_ms << "\n"
            << "min/p50/avg TFLOPS: " << tflops(min_ms) << " / "
            << tflops(p50_ms) << " / " << tflops(avg_ms) << "\n";

  // Check three output positions outside the timed region. Inputs are constant.
  const float expected = 0.5f * 0.25f * float(opt.k);
  for (int64_t index : {int64_t(0), elems_c / 2, elems_c - 1}) {
    __half sample;
    CUDA_CHECK(cudaMemcpy(&sample, c + index, sizeof(sample), cudaMemcpyDeviceToHost));
    float actual = __half2float(sample);
    if (!std::isfinite(actual) || std::fabs(actual - expected) > std::max(1.0f, expected * 0.01f)) {
      std::cerr << "GEMM output check failed: " << actual << " versus " << expected << "\n";
      return 1;
    }
  }
  std::cout << "TILESIGHT_METRIC tensor_gemm_fp16 " << tflops(p50_ms) << " TFLOP/s\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(a_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(b_desc));
  CUBLAS_CHECK(cublasLtMatrixLayoutDestroy(c_desc));
  CUBLAS_CHECK(cublasLtMatmulDescDestroy(op_desc));
  CUBLAS_CHECK(cublasLtDestroy(lt));
  CUDA_CHECK(cudaStreamDestroy(stream));
  if (workspace) CUDA_CHECK(cudaFree(workspace));
  CUDA_CHECK(cudaFree(a));
  CUDA_CHECK(cudaFree(b));
  CUDA_CHECK(cudaFree(c));
  return 0;
}
