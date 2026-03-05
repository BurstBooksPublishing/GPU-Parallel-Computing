#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

__device__ inline float heavy_work(float x) {
    #pragma unroll 64
    for (int i = 0; i < 64; ++i)
        x = fmaf(x, 1.0000001f, 0.000001f);
    return x;
}

__global__ void divergent_kernel(float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    if ((threadIdx.x & 1) == 0)
        out[tid] = heavy_work(tid * 1.0f);
    else
        out[tid] = tid * 1.0f;
}

__global__ void predicated_kernel(float *out, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float a = heavy_work(tid * 1.0f);
    float b = tid * 1.0f;
    bool p = ((threadIdx.x & 1) == 0);
    out[tid] = p ? a : b;
}

int main() {
    const int N = 1 << 20;
    const int BS = 128;
    int blocks = (N + BS - 1) / BS;
    float *d_out;
    cudaMalloc(&d_out, N * sizeof(float));
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    cudaEventRecord(s);
    divergent_kernel<<<blocks, BS>>>(d_out, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms_div = 0;
    cudaEventElapsedTime(&ms_div, s, e);

    cudaEventRecord(s);
    predicated_kernel<<<blocks, BS>>>(d_out, N);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms_pr = 0;
    cudaEventElapsedTime(&ms_pr, s, e);

    printf("Divergent kernel: %.3f ms\nPredicated kernel: %.3f ms\n", ms_div, ms_pr);

    cudaFree(d_out);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return 0;
}