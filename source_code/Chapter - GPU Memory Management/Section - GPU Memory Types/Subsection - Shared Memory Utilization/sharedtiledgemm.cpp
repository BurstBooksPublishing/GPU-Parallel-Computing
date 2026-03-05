#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

constexpr int TILE = 32;

__global__ void matmulTiled(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int N) {
    __shared__ float sA[TILE][TILE + 1];
    __shared__ float sB[TILE][TILE + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float acc = 0.0f;
    for (int m = 0; m < N; m += TILE) {
        sA[ty][tx] = (row < N && m + tx < N) ? A[row * N + (m + tx)] : 0.0f;
        sB[ty][tx] = (m + ty < N && col < N) ? B[(m + ty) * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) acc += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

int main() {
    const int N = 1024;
    const size_t bytes = size_t(N) * N * sizeof(float);

    float *hA, *hB, *hC;
    checkCuda(cudaMallocHost(&hA, bytes));
    checkCuda(cudaMallocHost(&hB, bytes));
    checkCuda(cudaMallocHost(&hC, bytes));

    for (int i = 0; i < N * N; ++i) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, bytes));
    checkCuda(cudaMalloc(&dB, bytes));
    checkCuda(cudaMalloc(&dC, bytes));

    checkCuda(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    matmulTiled<<<grid, block>>>(dA, dB, dC, N);

    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());

    checkCuda(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 10; ++i) assert(hC[i] == N);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}