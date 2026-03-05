#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

constexpr int TILE = 32;

__global__ void tiled_gemm(const float* __restrict__ A,
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
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    float *hA = static_cast<float*>(malloc(bytes));
    float *hB = static_cast<float*>(malloc(bytes));
    float *hC = static_cast<float*>(malloc(bytes));
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
    tiled_gemm<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seconds = ms * 1e-3;
    double gflops = (2.0 * static_cast<double>(N) * N * N) / (seconds * 1e9);
    printf("N=%d Time=%.3f ms GFLOPS=%.2f\n", N, ms, gflops);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}