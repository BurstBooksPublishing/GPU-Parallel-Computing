#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

__global__ void grid_stride_add(const float *x, const float *y, float *z, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = tid; i < N; i += stride) z[i] = x[i] + y[i];
}

__global__ void block_contiguous_add(const float *x, const float *y, float *z, size_t N) {
    size_t blocks = gridDim.x;
    size_t per_block = (N + blocks - 1) / blocks;
    size_t start = blockIdx.x * per_block;
    size_t end = min(start + per_block, N);
    for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) z[i] = x[i] + y[i];
}

int main() {
    const size_t N = 100000000;
    const size_t bytes = N * sizeof(float);
    float *h_x, *h_y, *h_z;
    CUDA_CHECK(cudaMallocHost(&h_x, bytes));
    CUDA_CHECK(cudaMallocHost(&h_y, bytes));
    CUDA_CHECK(cudaMallocHost(&h_z, bytes));
    for (size_t i = 0; i < N; ++i) { h_x[i] = 1.0f; h_y[i] = 2.0f; }

    float *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    grid_stride_add<<<blocks, threads>>>(d_x, d_y, d_z, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gbps = (double)N * sizeof(float) * 3.0 / (ms / 1000.0) / 1e9;
    printf("grid-stride: %.3f ms, %.3f GB/s\n", ms, gbps);

    CUDA_CHECK(cudaEventRecord(start));
    block_contiguous_add<<<blocks, threads>>>(d_x, d_y, d_z, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    gbps = (double)N * sizeof(float) * 3.0 / (ms / 1000.0) / 1e9;
    printf("block-contiguous: %.3f ms, %.3f GB/s\n", ms, gbps);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));
    CUDA_CHECK(cudaFreeHost(h_z));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}