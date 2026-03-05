#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>

template <int C>
__global__ void vecadd_coarsened(const float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float* __restrict__ c,
                                 size_t n) {
    size_t idx = blockIdx.x * blockDim.x * C + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < C; ++i) {
        size_t offset = idx + i * blockDim.x;
        if (offset < n) c[offset] = a[offset] + b[offset];
    }
}

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    const size_t N = 1ULL << 24;
    const int threads = 256;
    const int C = 4;
    const int blocks = (N + threads * C - 1) / (threads * C);

    float *hA, *hB, *hC;
    checkCuda(cudaMallocHost(&hA, N * sizeof(float)), "alloc hA");
    checkCuda(cudaMallocHost(&hB, N * sizeof(float)), "alloc hB");
    checkCuda(cudaMallocHost(&hC, N * sizeof(float)), "alloc hC");

    for (size_t i = 0; i < N; ++i) { hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, N * sizeof(float)), "alloc dA");
    checkCuda(cudaMalloc(&dB, N * sizeof(float)), "alloc dB");
    checkCuda(cudaMalloc(&dC, N * sizeof(float)), "alloc dC");

    checkCuda(cudaMemcpy(dA, hA, N * sizeof(float), cudaMemcpyHostToDevice), "H2D A");
    checkCuda(cudaMemcpy(dB, hB, N * sizeof(float), cudaMemcpyHostToDevice), "H2D B");

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);

    vecadd_coarsened<C><<<blocks, threads>>>(dA, dB, dC, N);
    checkCuda(cudaGetLastError(), "kernel launch");

    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms = 0;
    cudaEventElapsedTime(&ms, s, e);

    checkCuda(cudaMemcpy(hC, dC, N * sizeof(float), cudaMemcpyDeviceToHost), "D2H C");

    for (size_t i = 0; i < N; i += N / 16) assert(hC[i] == 3.0f);

    printf("N=%zu threads=%d C=%d time=%.3f ms  throughput=%.2f GB/s\n",
           N, threads, C, ms, 3.0f * N * sizeof(float) / (ms * 1e6));

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    return 0;
}