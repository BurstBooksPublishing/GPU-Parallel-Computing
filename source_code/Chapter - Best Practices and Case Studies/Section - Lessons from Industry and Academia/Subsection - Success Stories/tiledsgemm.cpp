#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <limits>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); if (e != cudaSuccess) { \
    std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    std::exit(EXIT_FAILURE); } } while(0)

constexpr int TILE = 32;

__global__ void sgemm_tiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N) {
  __shared__ float sA[TILE][TILE + 1]; // +1 avoids bank conflicts
  __shared__ float sB[TILE][TILE + 1];
  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
    int aCol = t * TILE + threadIdx.x;
    int bRow = t * TILE + threadIdx.y;
    sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
    sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    __syncthreads();
  }
  if (row < N && col < N) C[row * N + col] = acc;
}

int main(int argc, char* argv[]) {
  int N = (argc > 1) ? std::atoi(argv[1]) : 1024;
  if (N <= 0 || N % TILE) {
    std::cerr << "N must be positive and a multiple of " << TILE << "\n";
    return EXIT_FAILURE;
  }
  size_t bytes = size_t(N) * N * sizeof(float);
  std::vector<float> hA(N * N), hB(N * N), hC(N * N);
  std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  std::generate(hA.begin(), hA.end(), [&] { return dist(rng); });
  std::generate(hB.begin(), hB.end(), [&] { return dist(rng); });

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start));
  sgemm_tiled<<<grid, block>>>(dA, dB, dC, N);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
  double gflops = 2.0 * double(N) * N * N / (ms * 1e6);
  std::cout << std::fixed << std::setprecision(2)
            << "N=" << N << " time=" << ms << "ms GFLOPS=" << gflops << "\n";

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  return 0;
}