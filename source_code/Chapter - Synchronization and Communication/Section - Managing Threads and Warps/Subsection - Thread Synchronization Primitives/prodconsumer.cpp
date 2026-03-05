#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__device__ int ready_flag;
__device__ int data_array[1024];

__global__ void producer_consumer_kernel() {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (blockIdx.x == 0) {
        if (threadIdx.x < 32) {
            data_array[threadIdx.x] = threadIdx.x * 2;
        }
        __syncthreads();
        __threadfence();                 // ensure writes visible before flag set
        if (threadIdx.x == 0) {
            atomicExch(&ready_flag, 1);  // publish
        }
    } else {
        // exponential backoff spin-wait
        int wait = 1;
        while (atomicAdd(&ready_flag, 0) == 0) {
            for (int i = 0; i < wait; ++i) __nanosleep(0);
            wait = min(wait << 1, 1024);
        }
        __threadfence_block();           // acquire barrier
        if (threadIdx.x < 32) {
            const int val = data_array[threadIdx.x];
            if (threadIdx.x == 0) {
                printf("Consumer block %d observed data[0]=%d\n", blockIdx.x, val);
            }
        }
    }
}

int main() {
    int zero = 0;
    cudaMemcpyToSymbol(ready_flag, &zero, sizeof(int));

    producer_consumer_kernel<<<4, 128>>>();
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}