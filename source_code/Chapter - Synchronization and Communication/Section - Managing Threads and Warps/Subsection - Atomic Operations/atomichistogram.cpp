#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CHECK_CUDA(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); \
  } \
} while (0)

constexpr int BIN_COUNT = 256;
constexpr int BLOCK_SIZE = 256;

__global__ void hist_naive(const unsigned char *data, size_t n, int *global_hist) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  atomicAdd(&global_hist[data[idx]], 1);
}

__global__ void hist_block_private(const unsigned char *data, size_t n, int *global_hist) {
  extern __shared__ int s_hist[];
  for (int i = threadIdx.x; i < BIN_COUNT; i += blockDim.x) s_hist[i] = 0;
  __syncthreads();

  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    atomicAdd(&s_hist[data[i]], 1);
  }
  __syncthreads();

  for (int b = threadIdx.x; b < BIN_COUNT; b += blockDim.x) {
    int val = s_hist[b];
    if (val) atomicAdd(&global_hist[b], val);
  }
}

int main() {
  const size_t N = 1 << 24;
  unsigned char *data;
  int *hist;
  CHECK_CUDA(cudaMallocManaged(&data, N));
  CHECK_CUDA(cudaMallocManaged(&hist, BIN_COUNT * sizeof(int)));

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < N; ++i) data[i] = static_cast<unsigned char>(dist(rng));
  CHECK_CUDA(cudaMemset(hist, 0, BIN_COUNT * sizeof(int)));

  dim3 block(BLOCK_SIZE);
  dim3 grid((N + block.x - 1) / block.x);
  size_t shmem_bytes = BIN_COUNT * sizeof(int);

  hist_block_private<<<grid, block, shmem_bytes>>>(data, N, hist);
  CHECK_CUDA(cudaDeviceSynchronize());

  long long sum = 0;
  for (int i = 0; i < BIN_COUNT; ++i) sum += hist[i];
  printf("Total count = %lld (expected %zu)\n", sum, N);

  CHECK_CUDA(cudaFree(data));
  CHECK_CUDA(cudaFree(hist));
  return 0;
}