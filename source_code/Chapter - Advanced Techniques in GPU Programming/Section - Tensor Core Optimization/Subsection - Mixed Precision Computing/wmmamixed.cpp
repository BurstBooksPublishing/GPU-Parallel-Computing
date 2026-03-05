#include <cuda_runtime.h>
#include <mma.h>
#include <vector>
#include <iostream>
#include <cstdio>

using namespace nvcuda;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

static inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error " << e << " (" << cudaGetErrorString(e) << ")\n";
        std::exit(EXIT_FAILURE);
    }
}

__global__ void gemm(half const* A, half const* B, float* C,
                     int M, int N, int K) {
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = blockIdx.y * blockDim.y + threadIdx.y;

    int row = warpM * WMMA_M;
    int col = warpN * WMMA_N;
    if (row >= M || col >= N) return;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b;

        wmma::load_matrix_sync(a, A + row * K + k, K);
        wmma::load_matrix_sync(b, B + k * N + col, N);
        wmma::mma_sync(acc, a, b, acc);
    }

    wmma::store_matrix_sync(C + row * N + col, acc, N, wmma::mem_row_major);
}

int main() {
    const int M = 256, N = 256, K = 256;
    size_t sizeA = size_t(M) * K;
    size_t sizeB = size_t(K) * N;
    size_t sizeC = size_t(M) * N;

    std::vector<float> hA(sizeA, 1.0f), hB(sizeB, 1.0f), hC(sizeC);

    std::vector<half> hA_h(sizeA), hB_h(sizeB);
    std::transform(hA.begin(), hA.end(), hA_h.begin(), [](float f) { return __float2half(f); });
    std::transform(hB.begin(), hB.end(), hB_h.begin(), [](float f) { return __float2half(f); });

    half *dA, *dB;
    float *dC;
    checkCuda(cudaMalloc(&dA, sizeA * sizeof(half)));
    checkCuda(cudaMalloc(&dB, sizeB * sizeof(half)));
    checkCuda(cudaMalloc(&dC, sizeC * sizeof(float)));

    checkCuda(cudaMemcpy(dA, hA_h.data(), sizeA * sizeof(half), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dB, hB_h.data(), sizeB * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(128);
    dim3 grid((M + WMMA_M - 1) / WMMA_M, (N + WMMA_N - 1) / WMMA_N);
    gemm<<<grid, block>>>(dA, dB, dC, M, N, K);
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(hC.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "C[0] = " << hC[0] << '\n';

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}