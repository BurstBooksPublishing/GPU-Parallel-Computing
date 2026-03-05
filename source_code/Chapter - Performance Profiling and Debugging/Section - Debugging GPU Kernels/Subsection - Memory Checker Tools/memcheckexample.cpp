#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e, const char *msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void write_kernel(int *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = idx;
    }
}

int main() {
    const int N = 1024;
    int *d_data = nullptr;

    checkCuda(cudaMalloc(&d_data, N * sizeof(int)), "cudaMalloc failed");

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    write_kernel<<<blocks, threads>>>(d_data, N);

    checkCuda(cudaGetLastError(), "Kernel launch failed");
    checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");
    checkCuda(cudaFree(d_data), "cudaFree failed");

    std::puts("Completed host program");
    return 0;
}