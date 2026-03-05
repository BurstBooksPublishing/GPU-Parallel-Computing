#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                  \
  cudaError_t e = (call);                                      \
  if (e != cudaSuccess) {                                      \
    std::fprintf(stderr, "CUDA err %s:%d: %s\n",               \
                 __FILE__, __LINE__, cudaGetErrorString(e));   \
    std::exit(EXIT_FAILURE);                                   \
  }                                                            \
} while (0)

__global__ void add_kernel(const float* d_in, float* d_out, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d_out[i] = d_in[i] + 1.0f;
}

int main() {
  const size_t N = 1ULL << 24;
  const size_t bytes = N * sizeof(float);

  float* h_buf = nullptr;
  CUDA_CHECK(cudaMallocHost(&h_buf, bytes));

  for (size_t i = 0; i < N; ++i) h_buf[i] = static_cast<float>(i);

  float *d_in = nullptr, *d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_in, bytes));
  CUDA_CHECK(cudaMalloc(&d_out, bytes));

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, s));
  CUDA_CHECK(cudaMemcpyAsync(d_in, h_buf, bytes, cudaMemcpyHostToDevice, s));

  const int block = 256;
  const int grid = (static_cast<int>(N) + block - 1) / block;
  add_kernel<<<grid, block, 0, s>>>(d_in, d_out, N);

  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  std::printf("Async H2D + kernel elapsed: %.3f ms\n", ms);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFreeHost(h_buf));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(s));
  return 0;
}