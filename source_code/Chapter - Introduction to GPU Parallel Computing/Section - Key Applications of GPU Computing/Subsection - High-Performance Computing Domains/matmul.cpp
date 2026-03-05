#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); } } while (0)

constexpr int TILE = 32;

__global__ void matMulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N) {
  __shared__ float sA[TILE][TILE + 1]; // +1 to avoid bank conflicts
  __shared__ float sB[TILE][TILE + 1];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float sum = 0.0f;

  for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
    sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
  }

  if (row < N && col < N) C[row * N + col] = sum;
}

int main(int argc, char** argv) {
  int N = (argc > 1) ? std::atoi(argv[1]) : 2048;
  size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

  float *hA, *hB, *hC;
  CUDA_CHECK(cudaMallocHost(&hA, bytes));
  CUDA_CHECK(cudaMallocHost(&hB, bytes));
  CUDA_CHECK(cudaMallocHost(&hC, bytes));

  for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
    hA[i] = 1.0f;
    hB[i] = 1.0f;
  }

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));

  CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start));
  matMulTiled<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  double gflops = (2.0 * static_cast<double>(N) * N * N) / (ms * 1e6);
  printf("N=%d, time=%.3f ms, GFLOPS=%.2f\n", N, ms, gflops);

  CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaFreeHost(hA));
  CUDA_CHECK(cudaFreeHost(hB));
  CUDA_CHECK(cudaFreeHost(hC));
  return 0;
}