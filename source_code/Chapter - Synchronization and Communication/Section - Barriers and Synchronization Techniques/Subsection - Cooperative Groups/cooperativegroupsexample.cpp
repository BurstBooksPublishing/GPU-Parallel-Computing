#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do {                                           \
    cudaError_t e = (call);                                             \
    if (e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(e));             \
        cudaDeviceReset();                                              \
        exit(EXIT_FAILURE);                                             \
    }                                                                   \
} while (0)

__global__ void cooperative_phase_kernel(float *data, int N) {
    auto grid = cg::this_grid();
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float v = data[idx] * 2.0f;

    grid.sync();  // ensure all threads finish phase 1 before phase 2

    data[idx] = v + 1.0f;
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    int coopLaunch = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coopLaunch,
                                      cudaDevAttrCooperativeLaunch, dev));
    if (!coopLaunch) {
        fprintf(stderr, "Device does not support cooperative launch\n");
        return EXIT_FAILURE;
    }

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    if (blocks > prop.maxGridSize[0]) {
        fprintf(stderr, "Too many blocks for device\n");
        return EXIT_FAILURE;
    }

    // ensure occupancy allows cooperative launch
    int maxActive = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxActive, cooperative_phase_kernel, threads, 0));
    if (maxActive * prop.multiProcessorCount < blocks) {
        fprintf(stderr, "Not enough SMs for cooperative launch\n");
        return EXIT_FAILURE;
    }

    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, bytes));
    CUDA_CHECK(cudaMemset(d_data, 0, bytes));

    void *kernelArgs[] = { &d_data, &N };
    CUDA_CHECK(cudaLaunchCooperativeKernel(
        (void*)cooperative_phase_kernel, blocks, threads, kernelArgs));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_data));
    cudaDeviceReset();
    return 0;
}