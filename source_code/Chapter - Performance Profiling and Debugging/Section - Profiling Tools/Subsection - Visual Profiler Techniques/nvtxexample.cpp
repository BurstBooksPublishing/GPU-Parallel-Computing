#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

__global__ void vecAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

static void check(cudaError_t e, const char *m) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    float *hA, *hB, *hC;
    check(cudaMallocHost(&hA, bytes), "cudaMallocHost A");
    check(cudaMallocHost(&hB, bytes), "cudaMallocHost B");
    check(cudaMallocHost(&hC, bytes), "cudaMallocHost C");

    for (int i = 0; i < N; ++i) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    float *dA, *dB, *dC;
    check(cudaMalloc(&dA, bytes), "cudaMalloc A");
    check(cudaMalloc(&dB, bytes), "cudaMalloc B");
    check(cudaMalloc(&dC, bytes), "cudaMalloc C");

    nvtxRangePushA("H2D");
    check(cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice), "cudaMemcpyAsync A");
    check(cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice), "cudaMemcpyAsync B");
    nvtxRangePop();

    nvtxRangePushA("vecAdd");
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(dA, dB, dC, N);
    check(cudaGetLastError(), "kernel");
    nvtxRangePop();

    nvtxRangePushA("D2H");
    check(cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpyAsync C");
    check(cudaDeviceSynchronize(), "sync");
    nvtxRangePop();

    for (int i = 0; i < N; ++i) {
        if (hC[i] != 3.0f) {
            std::fprintf(stderr, "validation failed at %d\n", i);
            return EXIT_FAILURE;
        }
    }

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return EXIT_SUCCESS;
}