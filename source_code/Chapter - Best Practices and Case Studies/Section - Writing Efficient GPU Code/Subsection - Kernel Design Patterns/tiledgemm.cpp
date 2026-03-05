#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CALL(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  fprintf(stderr,"CUDA:%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

constexpr int TILE = 32; // tune per device: 16,32,64
__global__ void tiledGEMM(const float* __restrict__ A, const float* __restrict__ B,
                          float* __restrict__ C, int M, int N, int K) {
  __shared__ float sA[TILE][TILE];
  __shared__ float sB[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  float acc = 0.0f;
  for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    sA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
    sB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) C[row * N + col] = acc;
}

int main() {
  const int M = 1024, N = 1024, K = 1024;
  const size_t szA = size_t(M) * K * sizeof(float);
  const size_t szB = size_t(K) * N * sizeof(float);
  const size_t szC = size_t(M) * N * sizeof(float);

  float *hA = (float*)malloc(szA);
  float *hB = (float*)malloc(szB);
  float *hC = (float*)malloc(szC);
  for (size_t i = 0; i < M * K; ++i) hA[i] = 1.0f;
  for (size_t i = 0; i < K * N; ++i) hB[i] = 1.0f;

  float *dA, *dB, *dC;
  CUDA_CALL(cudaMalloc(&dA, szA));
  CUDA_CALL(cudaMalloc(&dB, szB));
  CUDA_CALL(cudaMalloc(&dC, szC));
  CUDA_CALL(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  cudaEvent_t s, e;
  CUDA_CALL(cudaEventCreate(&s));
  CUDA_CALL(cudaEventCreate(&e));
  CUDA_CALL(cudaEventRecord(s));
  tiledGEMM<<<grid, block>>>(dA, dB, dC, M, N, K);
  CUDA_CALL(cudaGetLastError());
  CUDA_CALL(cudaEventRecord(e));
  CUDA_CALL(cudaEventSynchronize(e));

  float ms;
  CUDA_CALL(cudaEventElapsedTime(&ms, s, e));
  printf("Elapsed: %f ms\n", ms);

  CUDA_CALL(cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaFree(dA));
  CUDA_CALL(cudaFree(dB));
  CUDA_CALL(cudaFree(dC));
  free(hA);
  free(hB);
  free(hC);
  return 0;
}