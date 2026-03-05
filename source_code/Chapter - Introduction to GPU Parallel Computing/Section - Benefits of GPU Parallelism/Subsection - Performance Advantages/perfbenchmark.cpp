#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                             \
      std::exit(EXIT_FAILURE);                                            \
    }                                                                     \
  } while (0)

__global__ void fma_bench_kernel(float* __restrict__ out, int iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  float x = 1.2345f + tid;
  #pragma unroll 8
  for (int i = 0; i < iterations; ++i) {
    x = fmaf(x, 1.0000003f, 0.9999997f);
    x = fmaf(x, 1.0000007f, 0.9999993f);
    x = fmaf(x, 0.9999999f, 1.0000001f);
    x = fmaf(x, 1.0000001f, 0.9999999f);
  }
  out[tid] = x;
}

int main(int argc, char* argv[]) {
  const int blocks  = 1024;
  const int threads = 256;
  const int iters   = (argc > 1) ? std::atoi(argv[1]) : 100000;
  const size_t n    = static_cast<size_t>(blocks) * threads;

  float* d_out;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  fma_bench_kernel<<<blocks, threads>>>(d_out, 1024);
  CUDA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  fma_bench_kernel<<<blocks, threads>>>(d_out, iters);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  const double flops_per_thread = static_cast<double>(iters) * 4.0 * 2.0;
  const double total_flops      = flops_per_thread * n;
  const double gflops           = total_flops / (ms * 1e6);

  std::printf("Threads: %zu, Iterations/thread: %d, Time: %.3f ms, GFLOPS: %.2f\n",
              n, iters, ms, gflops);

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}