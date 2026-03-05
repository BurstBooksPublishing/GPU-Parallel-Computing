#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t e = (call);                                           \
        if (e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                               \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

__global__ void wmma_gemm_kernel(const half *A, const half *B, float *C,
                                 int M, int N, int K) {
    int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int tilesPerRow = N / WMMA_N;
    int m_tile = warpId / tilesPerRow;
    int n_tile = warpId % tilesPerRow;
    if (m_tile >= M / WMMA_M) return;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k_tile = 0; k_tile < K / WMMA_K; ++k_tile) {
        const half *a_tile = A + (m_tile * WMMA_M) * K + k_tile * WMMA_K;
        const half *b_tile = B + (k_tile * WMMA_K) * N + n_tile * WMMA_N;
        wmma::load_matrix_sync(a_frag, a_tile, K);
        wmma::load_matrix_sync(b_frag, b_tile, N);
        wmma::mma_sync(acc, a_frag, b_frag, acc);
    }
    float *c_tile = C + (m_tile * WMMA_M) * N + n_tile * WMMA_N;
    wmma::store_matrix_sync(c_tile, acc, N, wmma::mem_row_major);
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(float);

    half *hA = (half *)malloc(sizeA);
    half *hB = (half *)malloc(sizeB);
    float *hC = (float *)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) hA[i] = __float2half(0.01f * (i % 7));
    for (int i = 0; i < K * N; ++i) hB[i] = __float2half(0.02f * (i % 5));

    half *dA, *dB;
    float *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeA));
    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dC, sizeC));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));

    int warpsPerBlock = 8;
    int threadsPerBlock = warpsPerBlock * warpSize;
    int totalWarps = (M / WMMA_M) * (N / WMMA_N);
    int blocks = (totalWarps + warpsPerBlock - 1) / warpsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    wmma_gemm_kernel<<<blocks, threadsPerBlock>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gflops = 2.0 * (double)M * N * K / (ms * 1e6);
    printf("Runtime: %.3f ms, GFLOPS: %.2f\n", ms, gflops);

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    return 0;
}