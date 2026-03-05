#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t e = (call);                                               \
    if (e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(e));                                     \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

__global__ void heavy_kernel(float *out, int n, int work_iters) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;

  float x = static_cast<float>(idx);
  #pragma unroll 8
  for (int i = 0; i < work_iters; ++i)
    x = fmaf(x, 1.0000001f, 0.0000001f);
  out[idx] = x;
}

int main(int argc, char **argv) {
  int n = 1 << 20;
  int work_iters = 1000;
  int serial_iters = 1000000;

  if (argc > 1) work_iters = std::atoi(argv[1]);
  if (argc > 2) serial_iters = std::atoi(argv[2]);

  float *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

  auto t0 = std::chrono::high_resolution_clock::now();
  volatile double acc = 0.0;
  for (int i = 0; i < serial_iters; ++i) acc += i * 1e-12;
  auto t1 = std::chrono::high_resolution_clock::now();
  double host_serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  cudaEvent_t e_start, e_stop;
  CUDA_CHECK(cudaEventCreate(&e_start));
  CUDA_CHECK(cudaEventCreate(&e_stop));

  const int block = 256;
  const int grid = (n + block - 1) / block;

  CUDA_CHECK(cudaEventRecord(e_start));
  heavy_kernel<<<grid, block>>>(d_out, n, work_iters);
  CUDA_CHECK(cudaEventRecord(e_stop));
  CUDA_CHECK(cudaEventSynchronize(e_stop));

  float kernel_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, e_start, e_stop));

  printf("serial_ms=%.3f, kernel_ms=%.3f, total_ms=%.3f\n",
         host_serial_ms, kernel_ms, host_serial_ms + kernel_ms);

  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaEventDestroy(e_start));
  CUDA_CHECK(cudaEventDestroy(e_stop));
  return 0;
}