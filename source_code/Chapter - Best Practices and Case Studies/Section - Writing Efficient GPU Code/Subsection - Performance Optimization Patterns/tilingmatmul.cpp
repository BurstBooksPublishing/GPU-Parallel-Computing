#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                              \
    do {                                                                         \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

constexpr int TILE = 32;

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
    const int N = 2048;
    const size_t bytes = N * N * sizeof(float);

    float *hA, *hB, *hC;
    CHECK(cudaMallocHost(&hA, bytes));
    CHECK(cudaMallocHost(&hB, bytes));
    CHECK(cudaMallocHost(&hC, bytes));

    for (size_t i = 0; i < size_t(N) * N; ++i) {
        hA[i] = 1.0f;
        hB[i] = 1.0f;
    }

    float *dA, *dB, *dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));

    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    CHECK(cudaGetLastError());
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Tiled matmul %d x %d: %.3f ms\n", N, N, ms);

    CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    return 0;
}