#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; std::exit(EXIT_FAILURE); } } while(0)

__global__ void vecAdd(const float* a, const float* b, float* c, size_t N) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride) c[i] = a[i] + b[i];
}

dim3 computeGrid(size_t N, int threadsPerBlock, int elemsPerThread, int minBlocksPerSM) {
    int dev; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    size_t blocksNeeded = (N + (size_t)threadsPerBlock * elemsPerThread - 1)
                          / ((size_t)threadsPerBlock * elemsPerThread);
    size_t minBlocks = (size_t)minBlocksPerSM * prop.multiProcessorCount;
    blocksNeeded = std::max(blocksNeeded, minBlocks);

    int maxX = prop.maxGridSize[0];
    int maxY = prop.maxGridSize[1];
    dim3 grid;
    grid.x = static_cast<int>(std::min(blocksNeeded, (size_t)maxX));
    size_t rem = (blocksNeeded + maxX - 1) / maxX;
    grid.y = static_cast<int>(std::min(rem, (size_t)maxY));
    grid.z = 1;
    return grid;
}

int main() {
    const size_t N = 50'000'000;
    const int threadsPerBlock = 256;
    const int elemsPerThread = 1;
    const int minBlocksPerSM = 6;

    dim3 grid = computeGrid(N, threadsPerBlock, elemsPerThread, minBlocksPerSM);

    std::vector<float> hA(N, 1.0f), hB(N, 2.0f), hC(N);
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(s));
    vecAdd<<<grid, threadsPerBlock>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));

    double bytes = static_cast<double>(N) * 3.0 * sizeof(float);
    double gbps = bytes / (ms * 1e-3) / 1e9;
    std::cout << "Grid (" << grid.x << "," << grid.y << "), time " << ms << " ms, bandwidth " << gbps << " GB/s\n";

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}