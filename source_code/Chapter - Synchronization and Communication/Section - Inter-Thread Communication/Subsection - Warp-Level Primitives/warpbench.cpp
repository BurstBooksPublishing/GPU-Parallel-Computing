#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ inline float warpReduceShfl(float val, unsigned fullMask) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(fullMask, val, offset);
    return val; // valid in lane 0
}

__global__ void warpReductionsKernel(const float* __restrict__ in,
                                     float* __restrict__ out_shfl,
                                     float* __restrict__ out_sm,
                                     int N) {
    extern __shared__ float sdata[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (warpSize - 1);
    int warpId = threadIdx.x / warpSize;

    float val = (tid < N) ? in[tid] : 0.0f;

    // shuffle reduction
    float sum_shfl = warpReduceShfl(val, 0xffffffffu);
    if (lane == 0)
        out_shfl[blockIdx.x * (blockDim.x / warpSize) + warpId] = sum_shfl;

    // shared-memory reduction
    int smIndex = warpId * warpSize + lane;
    sdata[smIndex] = val;
    __syncthreads();

    if (lane < 16) sdata[smIndex] += sdata[smIndex + 16];
    if (lane < 8)  sdata[smIndex] += sdata[smIndex + 8];
    if (lane < 4)  sdata[smIndex] += sdata[smIndex + 4];
    if (lane < 2)  sdata[smIndex] += sdata[smIndex + 2];
    if (lane == 0)
        out_sm[blockIdx.x * (blockDim.x / warpSize) + warpId] = sdata[smIndex];
}