#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    std::exit(EXIT_FAILURE); \
  } \
} while(0)

constexpr int TILE = 16;

__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
  __shared__ float sA[TILE][TILE];
  __shared__ float sB[TILE][TILE];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = by * TILE + ty;
  int col = bx * TILE + tx;

  float acc = 0.0f;
  for (int m = 0; m < (N + TILE - 1) / TILE; ++m) {
    int aCol = m * TILE + tx;
    int bRow = m * TILE + ty;
    sA[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
    sB[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = acc;
}

int main() {
  constexpr int N = 1024;
  const size_t bytes = N * N * sizeof(float);

  float *hA, *hB, *hC;
  CUDA_CHECK(cudaMallocHost(&hA, bytes));
  CUDA_CHECK(cudaMallocHost(&hB, bytes));
  CUDA_CHECK(cudaMallocHost(&hC, bytes));

  for (int i = 0; i < N * N; ++i) {
    hA[i] = static_cast<float>(i % 100);
    hB[i] = static_cast<float>((i + 7) % 100);
  }

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  const double gflops = (2.0 * static_cast<double>(N) * N * N) / (ms * 1e6);
  std::printf("N=%d time=%.3f ms GFLOPS=%.2f\n", N, ms, gflops);

  CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

  double gold = 0.0;
  constexpr int r = 123, c = 456;
  for (int k = 0; k < N; ++k) gold += hA[r * N + k] * hB[k * N + c];
  const double diff = std::abs(gold - hC[r * N + c]);
  std::printf("Validation diff=%.6e\n", diff);

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFreeHost(hA));
  CUDA_CHECK(cudaFreeHost(hB));
  CUDA_CHECK(cudaFreeHost(hC));
  return 0;
}