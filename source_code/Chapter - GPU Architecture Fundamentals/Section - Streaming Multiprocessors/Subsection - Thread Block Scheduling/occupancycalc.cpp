#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t e = (call);                                           \
        if (e != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                \
                         __FILE__, __LINE__, cudaGetErrorString(e));      \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

__global__ void example_kernel(float* a) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1) a[idx] += 0.0f;
}

int main() {
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    cudaFuncAttributes attrs;
    CUDA_CHECK(cudaFuncGetAttributes(&attrs, example_kernel));

    const int threadsPerBlock   = 256;
    const int dynamicSmemBytes  = 0;
    const int regsPerThread     = attrs.numRegs;
    const int staticSmemPerBlock= attrs.sharedSizeBytes;

    const int regsPerSM         = prop.regsPerMultiprocessor;
    const int smemPerSM         = prop.sharedMemPerMultiprocessor;
    const int maxThreadsPerSM   = prop.maxThreadsPerMultiProcessor;
    const int warpSize          = prop.warpSize;
    const int maxWarpsPerSM     = maxThreadsPerSM / warpSize;

    const int maxByRegs   = regsPerThread   ? regsPerSM   / (regsPerThread   * threadsPerBlock) : 0;
    const int maxBySmem   = staticSmemPerBlock ? smemPerSM / (staticSmemPerBlock + dynamicSmemBytes) : 0;
    const int maxByThreads= maxThreadsPerSM / threadsPerBlock;
    const int hwMaxBlocksPerSM = 32;

    int residentBlocks = maxByRegs;
    residentBlocks = residentBlocks > maxBySmem   ? maxBySmem   : residentBlocks;
    residentBlocks = residentBlocks > maxByThreads? maxByThreads: residentBlocks;
    residentBlocks = residentBlocks > hwMaxBlocksPerSM ? hwMaxBlocksPerSM : residentBlocks;

    const int activeWarps = residentBlocks * (threadsPerBlock / warpSize);
    const double manualOccupancy = static_cast<double>(activeWarps) / maxWarpsPerSM;

    int occupancyBlocks = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &occupancyBlocks, example_kernel, threadsPerBlock, dynamicSmemBytes));
    const double apiOccupancy = static_cast<double>(occupancyBlocks * (threadsPerBlock / warpSize)) / maxWarpsPerSM;

    std::printf("Device: %s\n", prop.name);
    std::printf("Regs/SM=%d, Smem/SM=%d, MaxThreads/SM=%d, WarpSize=%d\n",
                regsPerSM, smemPerSM, maxThreadsPerSM, warpSize);
    std::printf("Kernel regs/thread=%d, static smem/block=%d\n", regsPerThread, staticSmemPerBlock);
    std::printf("Manual: residentBlocks=%d, occupancy=%.2f\n", residentBlocks, manualOccupancy);
    std::printf("API:    residentBlocks=%d, occupancy=%.2f\n", occupancyBlocks, apiOccupancy);

    return 0;
}