#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cuda_runtime.h>

__global__ void dummyKernel(float *data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) data[0] = 0.0f;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: %s <threadsPerBlock> <regsPerThread>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const int threadsPerBlock = std::atoi(argv[1]);
    const int regsPerThread   = std::atoi(argv[2]);

    if (threadsPerBlock <= 0 || regsPerThread <= 0) {
        std::fprintf(stderr, "Both arguments must be positive integers.\n");
        return EXIT_FAILURE;
    }

    int dev = 0;
    cudaDeviceProp prop{};
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);

    const int maxBlocksByRegs    = prop.regsPerMultiprocessor / (regsPerThread * threadsPerBlock);
    const int maxBlocksByThreads = prop.maxThreadsPerMultiProcessor / threadsPerBlock;
    const int maxBlocksHW        = prop.maxBlocksPerMultiProcessor;

    const int blocksPerSM     = std::min({maxBlocksByRegs, maxBlocksByThreads, maxBlocksHW});
    const int residentThreads = blocksPerSM * threadsPerBlock;
    const double occupancy    = static_cast<double>(residentThreads) /
                                static_cast<double>(prop.maxThreadsPerMultiProcessor);

    int occBlocksApi = 0;
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occBlocksApi, reinterpret_cast<const void*>(dummyKernel), threadsPerBlock, 0);
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    const int occThreadsApi = occBlocksApi * threadsPerBlock;
    const double occupancyApi = static_cast<double>(occThreadsApi) /
                                static_cast<double>(prop.maxThreadsPerMultiProcessor);

    std::printf("Device: %s\n", prop.name);
    std::printf("Registers/SM: %d, MaxThreads/SM: %d, MaxBlocks/SM: %d\n",
                prop.regsPerMultiprocessor,
                prop.maxThreadsPerMultiProcessor,
                prop.maxBlocksPerMultiProcessor);
    std::printf("Input: threadsPerBlock=%d, regsPerThread=%d\n",
                threadsPerBlock, regsPerThread);
    std::printf("Calc blocks/SM=%d, resident threads=%d, occupancy=%.2f\n",
                blocksPerSM, residentThreads, occupancy);
    std::printf("CUDA API blocks/SM=%d, resident threads=%d, occupancy=%.2f\n",
                occBlocksApi, occThreadsApi, occupancyApi);

    return EXIT_SUCCESS;
}