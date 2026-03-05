#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(EXIT_FAILURE); }} while(0)

__global__ void staged_work_kernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) data[idx] = idx * 1.0f;

    __threadfence(); // flush writes before grid sync

    cg::grid_group grid = cg::this_grid();
    grid.sync(); // requires cooperative launch

    if (idx < N) data[idx] += 1.0f;
}

__global__ void phase_kernel(float *data, int N, int phase) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    if (phase == 1) data[idx] = idx * 1.0f;
    else data[idx] += 1.0f;
}

int main() {
    const int N = 1 << 20;
    float *d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const bool coop = prop.cooperativeLaunch;

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    if (coop) {
        void *args[] = { &d_data, &N };
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)staged_work_kernel, grid, block, args));
    } else {
        phase_kernel<<<grid, block>>>(d_data, N, 1);
        CUDA_CHECK(cudaDeviceSynchronize());
        phase_kernel<<<grid, block>>>(d_data, N, 2);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
    return 0;
}