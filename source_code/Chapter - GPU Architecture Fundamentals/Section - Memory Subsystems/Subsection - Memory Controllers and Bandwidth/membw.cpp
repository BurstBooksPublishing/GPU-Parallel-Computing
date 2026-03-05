#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error " << cudaGetErrorString(err)         \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';   \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

__global__ void stream_store(float* __restrict__ out, size_t n) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = idx; i < n; i += stride)
        out[i] = __int2float_rn(static_cast<int>(i));
}

int main() {
    const size_t TEST_BYTES = size_t(512) << 20;          // 512 MiB
    const size_t TEST_FLOATS = TEST_BYTES / sizeof(float);

    float *dA = nullptr, *dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, TEST_BYTES));
    CUDA_CHECK(cudaMalloc(&dB, TEST_BYTES));

    CUDA_CHECK(cudaMemset(dA, 0, TEST_BYTES));
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUDA_CHECK(cudaMemcpy(dB, dA, TEST_BYTES, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gbps = (TEST_BYTES / 1e9) / (ms / 1e3);
    std::cout << "D2D memcpy: " << gbps << " GB/s (" << ms << " ms)\n";

    const int threads = 256;
    int blocks = (TEST_FLOATS + threads - 1) / threads;
    blocks = std::min(blocks, 65535);

    CUDA_CHECK(cudaEventRecord(start));
    stream_store<<<blocks, threads>>>(dA, TEST_FLOATS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    gbps = (TEST_BYTES / 1e9) / (ms / 1e3);
    std::cout << "Streaming store: " << gbps << " GB/s (" << ms << " ms)\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}