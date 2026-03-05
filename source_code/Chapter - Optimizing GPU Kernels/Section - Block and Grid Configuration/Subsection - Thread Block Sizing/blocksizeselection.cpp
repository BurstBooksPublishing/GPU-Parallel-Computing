#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <iomanip>

#define CUDA_CHECK(call) do {                                           \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(e)); } \
} while(0)

__global__ void vecAdd(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const std::size_t N = 1 << 24;
    std::vector<float> hA(N, 1.23f), hB(N, 4.56f), hC(N);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int warpSize = prop.warpSize;
    std::cout << "Device: " << prop.name << "\n";

    float bestMs = std::numeric_limits<float>::max();
    int bestBlock = 0;

    for (int block = warpSize; block <= 1024; block += warpSize) {
        int blocksPerSM = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, vecAdd, block, 0));
        float occupancy = (blocksPerSM * block) / static_cast<float>(prop.maxThreadsPerMultiProcessor);
        int grid = static_cast<int>((N + block - 1) / block);

        vecAdd<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t s, e;
        CUDA_CHECK(cudaEventCreate(&s));
        CUDA_CHECK(cudaEventCreate(&e));
        CUDA_CHECK(cudaEventRecord(s));
        const int iters = 50;
        for (int i = 0; i < iters; ++i) vecAdd<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaEventRecord(e));
        CUDA_CHECK(cudaEventSynchronize(e));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
        ms /= iters;
        CUDA_CHECK(cudaEventDestroy(s));
        CUDA_CHECK(cudaEventDestroy(e));

        double throughputGBs = (N * sizeof(float) * 3) / (ms * 1e6);
        std::cout << "block=" << block << " occupancy=" << std::fixed << std::setprecision(2) << occupancy
                  << " avg_ms=" << ms << " throughput(GB/s)=" << throughputGBs << "\n";

        if (ms < bestMs) {
            bestMs = ms;
            bestBlock = block;
        }
    }

    std::cout << "Best block size: " << bestBlock << " avg_ms=" << bestMs << "\n";
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}