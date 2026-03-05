#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

constexpr int BLOCK_SIZE = 256;

__global__ __launch_bounds__(BLOCK_SIZE, 2)
void tiled_dot(const float* __restrict__ a,
               const float* __restrict__ b,
               float* __restrict__ out,
               size_t N) {
    extern __shared__ float s[];                 // dynamic shared memory
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = threadIdx.x;
    float acc = 0.0f;

    for (size_t tile = blockIdx.x * blockDim.x; tile < N; tile += gridDim.x * blockDim.x) {
        int idx = tile + lane;
        s[lane] = (idx < N) ? a[idx] : 0.0f;     // load tile to shared
        __syncthreads();

        for (int i = 0; i < blockDim.x && tile + i < N; ++i)
            acc += s[i] * b[tile + i];           // compute partial dot
        __syncthreads();
    }

    if (tid < N) out[tid] = acc;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int minGridSize, gridSize;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &gridSize, tiled_dot, 0, BLOCK_SIZE));

    printf("Registers/SM: %d, Max threads/SM: %d\n",
           prop.regsPerMultiprocessor, prop.maxThreadsPerMultiProcessor);
    printf("Suggested grid size: %d (minGridSize=%d)\n", gridSize, minGridSize);

    return 0;
}