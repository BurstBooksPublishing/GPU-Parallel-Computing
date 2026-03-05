#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(ans) do { cudaError_t e = (ans); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void process_coarse(const float* __restrict__ in, float* __restrict__ out,
                               size_t N, int grain) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  size_t start = tid * static_cast<size_t>(grain);
  size_t end = (start + grain < N) ? start + grain : N;
  if (start >= N) return;

  float acc = 0.0f;
  for (size_t i = start; i < end; ++i) {
    float v = in[i];
    acc = acc * 1.0000001f + v;
  }
  out[tid] = acc;
}

void launch_with_grain(const float* d_in, float* d_out, size_t N, int grain) {
  int blockSize = 0, minGridSize = 0;
  CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                       reinterpret_cast<const void*>(process_coarse), 0, 0));
  size_t threads_needed = (N + grain - 1) / grain;
  int grid = static_cast<int>((threads_needed + blockSize - 1) / blockSize);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  process_coarse<<<grid, blockSize>>>(d_in, d_out, N, grain);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  printf("grain=%d grid=%d block=%d time=%.3f ms\n", grain, grid, blockSize, ms);
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}