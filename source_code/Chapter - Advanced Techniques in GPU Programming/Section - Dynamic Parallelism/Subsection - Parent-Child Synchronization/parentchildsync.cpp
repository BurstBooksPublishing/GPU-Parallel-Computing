#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__global__ void childKernel(int *counter, int work_iters) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;
    for (int i = 0; i < work_iters; ++i) acc += sinf(gid + i);
    __threadfence_system();
    atomicSub(counter, 1);
}

__global__ void parentKernel(int *counter, int tasks, int child_blocks, int child_threads, int work_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= tasks) return;
    atomicAdd(counter, 1);
    childKernel<<<child_blocks, child_threads>>>(counter, work_iters);
    int val, backoff = 1;
    do {
        val = atomicAdd(counter, 0);
        if (val) {
            for (volatile int i = 0; i < backoff; ++i);
            backoff = min(backoff << 1, 1024);
        }
    } while (val);
}

int main() {
    const int tasks = 1024;
    const int child_blocks = 4;
    const int child_threads = 128;
    const int work_iters = 1000;

    int *d_counter;
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    parentKernel<<<(tasks + 255) / 256, 256>>>(d_counter, tasks, child_blocks, child_threads, work_iters);
    cudaDeviceSynchronize();

    cudaFree(d_counter);
    printf("Parent–child synchronization run complete.\n");
    return 0;
}