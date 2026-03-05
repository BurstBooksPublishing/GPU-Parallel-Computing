#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t e = (call);                                               \
    if (e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
              cudaGetErrorString(e));                                     \
      exit(EXIT_FAILURE);                                                 \
    }                                                                     \
  } while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void wmma_gemm_fp16_fp32(const half *A, const half *B, float *C,
                                    int M, int N, int K, int lda, int ldb,
                                    int ldc) {
  int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int tilesX = N / WMMA_N;
  int tileY = warpId / tilesX;
  int tileX = warpId % tilesX;
  if (tileY * WMMA_M >= M || tileX * WMMA_N >= N) return;

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);

  for (int k = 0; k < K; k += WMMA_K) {
    const half *a_tile = A + (tileY * WMMA_M) * lda + k;
    const half *b_tile = B + k * ldb + (tileX * WMMA_N);
    wmma::load_matrix_sync(a_frag, a_tile, lda);
    wmma::load_matrix_sync(b_frag, b_tile, ldb);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  float *c_tile = C + (tileY * WMMA_M) * ldc + (tileX * WMMA_N);
  wmma::store_matrix_sync(c_tile, c_frag, ldc, wmma::mem_row_major);
}

int main() {
  const int M = 1024;
  const int N = 1024;
  const int K = 1024;
  const int lda = K;
  const int ldb = N;
  const int ldc = N;

  size_t sizeA = static_cast<size_t>(M) * K * sizeof(half);
  size_t sizeB = static_cast<size_t>(K) * N * sizeof(half);
  size_t sizeC = static_cast<size_t>(M) * N * sizeof(float);

  half *dA;
  half *dB;
  float *dC;
  CUDA_CHECK(cudaMalloc(&dA, sizeA));
  CUDA_CHECK(cudaMalloc(&dB, sizeB));
  CUDA_CHECK(cudaMalloc(&dC, sizeC));

  int totalWarps = (M / WMMA_M) * (N / WMMA_N);
  int threadsPerBlock = 128;
  int warpsPerBlock = threadsPerBlock / warpSize;
  int blocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

  wmma_gemm_fp16_fp32<<<blocks, threadsPerBlock>>>(dA, dB, dC, M, N, K, lda,
                                                   ldb, ldc);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}