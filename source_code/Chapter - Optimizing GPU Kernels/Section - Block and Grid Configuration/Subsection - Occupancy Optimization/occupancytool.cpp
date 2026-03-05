#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",              \
                         cudaGetErrorString(err), __FILE__, __LINE__);    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

__global__ void dummy_kernel(float* __restrict__) {}   // occupancy probe only

int main() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int warpSize          = prop.warpSize;
    const int maxThreadsPerSM   = prop.maxThreadsPerMultiProcessor;
    const int maxWarpsPerSM     = maxThreadsPerSM / warpSize;

    std::cout << "Device: " << prop.name
              << ", warpSize=" << warpSize
              << ", maxThreadsPerSM=" << maxThreadsPerSM << '\n';

    std::vector<int> blockSizes;
    for (int b = warpSize; b <= prop.maxThreadsPerBlock; b += warpSize)
        blockSizes.push_back(b);

    std::cout << "blockSize,activeBlocksPerSM,activeWarpsPerSM,occupancy(%)\n";

    for (int bs : blockSizes) {
        int activeBlocks = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                       &activeBlocks,
                       reinterpret_cast<const void*>(dummy_kernel),
                       bs, 0));

        int activeWarps = activeBlocks * ((bs + warpSize - 1) / warpSize);
        double occupancy = 100.0 * activeWarps / maxWarpsPerSM;

        std::cout << bs << ',' << activeBlocks << ','
                  << activeWarps << ',' << occupancy << '\n';
    }
    return 0;
}