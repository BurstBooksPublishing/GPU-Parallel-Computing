#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE 32

#define CUDA_CALL(cmd) do { cudaError_t e = (cmd); if (e != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

__global__ void tiled_gemm(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int N) {
  __shared__ float sA[TILE][TILE + 1]; // +1 to avoid bank conflicts
  __shared__ float sB[TILE][TILE + 1];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;
  int row = by * TILE + ty;
  int col = bx * TILE + tx;

  float acc = 0.0f;
  for (int m = 0; m < N; m += TILE) {
    sA[ty][tx] = (row < N && (m + tx) < N) ? A[row * N + m + tx] : 0.0f;
    sB[ty][tx] = ((m + ty) < N && col < N) ? B[(m + ty) * N + col] : 0.0f;
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += sA[ty][k] * sB[k][tx];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = acc;
}

int main() {
  const int N = 4096;
  const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

  float *hA, *hB, *hC;
  CUDA_CALL(cudaMallocHost(&hA, bytes));
  CUDA_CALL(cudaMallocHost(&hB, bytes));
  CUDA_CALL(cudaMallocHost(&hC, bytes));

  for (size_t i = 0; i < static_cast<size_t>(N) * N; ++i) {
    hA[i] = 1.0f;
    hB[i] = 1.0f;
  }

  float *dA, *dB, *dC;
  CUDA_CALL(cudaMalloc(&dA, bytes));
  CUDA_CALL(cudaMalloc(&dB, bytes));
  CUDA_CALL(cudaMalloc(&dC, bytes));

  CUDA_CALL(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
  tiled_gemm<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaDeviceSynchronize());

  CUDA_CALL(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

  const float expected = static_cast<float>(N);
  for (int i = 0; i < N; i += N / 16)
    if (abs(hC[i * N + i] - expected) > 1e-3f) {
      fprintf(stderr, "Validation failed\n");
      return 1;
    }

  printf("OK\n");

  CUDA_CALL(cudaFree(dA));
  CUDA_CALL(cudaFree(dB));
  CUDA_CALL(cudaFree(dC));
  CUDA_CALL(cudaFreeHost(hA));
  CUDA_CALL(cudaFreeHost(hB));
  CUDA_CALL(cudaFreeHost(hC));
  return 0;
}