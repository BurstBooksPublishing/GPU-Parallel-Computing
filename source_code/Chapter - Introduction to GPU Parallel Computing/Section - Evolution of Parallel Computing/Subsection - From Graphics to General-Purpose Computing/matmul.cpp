#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                         \
    cudaError_t e = (call);                                           \
    if (e != cudaSuccess) {                                           \
        fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,    \
                cudaGetErrorString(e));                               \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while(0)

constexpr int BLOCK = 32;

__global__ void matmul_tiled(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N) {
    __shared__ float sA[BLOCK][BLOCK];
    __shared__ float sB[BLOCK][BLOCK];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK + ty;
    int col = bx * BLOCK + tx;

    float acc = 0.0f;
    for (int m = 0; m < (N + BLOCK - 1) / BLOCK; ++m) {
        int a_col = m * BLOCK + tx;
        int b_row = m * BLOCK + ty;
        sA[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;
        sB[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK; ++k) acc += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = acc;
}

int main() {
    const int N = 2048;
    const size_t bytes = size_t(N) * N * sizeof(float);

    float *hA, *hB, *hC;
    CUDA_CHECK(cudaMallocHost(&hA, bytes));
    CUDA_CHECK(cudaMallocHost(&hB, bytes));
    CUDA_CHECK(cudaMallocHost(&hC, bytes));

    for (size_t i = 0; i < size_t(N) * N; ++i) { hA[i] = 1.0f; hB[i] = 1.0f; }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (N + BLOCK - 1) / BLOCK);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    matmul_tiled<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gflops = 2.0 * double(N) * N * N / (ms * 1e6);
    printf("N=%d, time=%.3f ms, %.2f GFLOPS\n", N, ms, gflops);

    CUDA_CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < size_t(N) * N; ++i) {
        if (std::abs(hC[i] - N) > 1e-3) { fprintf(stderr, "Mismatch\n"); break; }
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));
    return 0;
}