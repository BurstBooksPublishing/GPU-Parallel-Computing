#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t e = (call);                                           \
        if (e != cudaSuccess) {                                           \
            std::cerr << "CUDA error " << e << " at " << __FILE__ << ":" \
                      << __LINE__ << " - " << cudaGetErrorString(e)      \
                      << std::endl;                                       \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

__global__ void dummy_kernel() {} // no-op kernel for occupancy query

int main(int argc, char** argv) {
    int dev = 0;
    CHECK_CUDA(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    int blockSize = 256;
    if (argc > 1) blockSize = std::atoi(argv[1]);
    if (blockSize <= 0 || blockSize > prop.maxThreadsPerBlock) {
        std::cerr << "Invalid block size\n";
        return EXIT_FAILURE;
    }

    int blocksPerSM = 0;
    CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocksPerSM, dummy_kernel, blockSize, 0));

    int maxWarpsPerSM = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int activeWarps = (blocksPerSM * blockSize) / prop.warpSize;
    double occupancy = static_cast<double>(activeWarps) / maxWarpsPerSM;

    std::cout << "Device: " << prop.name << '\n'
              << "Block size: " << blockSize << '\n'
              << "Blocks per SM: " << blocksPerSM << '\n'
              << "Active warps per SM: " << activeWarps << " / " << maxWarpsPerSM
              << "  (occupancy = " << occupancy * 100.0 << "%)\n"
              << "Warp size: " << prop.warpSize
              << ", max threads/SM: " << prop.maxThreadsPerMultiProcessor << '\n';
    return 0;
}