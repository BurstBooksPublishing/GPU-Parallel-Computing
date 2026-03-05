#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CALL(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); } } while(0)

__global__ void saxpy_kernel(const float a, const float *x, float *y, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) y[i] = a * x[i] + y[i];
}

int main() {
    const size_t N = 1ULL << 26;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    const float a = 2.5f;

    CUDA_CALL(cudaSetDevice(0));
    float *d_x, *d_y;
    CUDA_CALL(cudaMalloc(&d_x, N * sizeof(float)));
    CUDA_CALL(cudaMalloc(&d_y, N * sizeof(float)));

    // initialize arrays
    CUDA_CALL(cudaMemset(d_x, 1, N * sizeof(float)));
    CUDA_CALL(cudaMemset(d_y, 2, N * sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));

    const int repeats = 10;
    CUDA_CALL(cudaDeviceSynchronize());
    CUDA_CALL(cudaEventRecord(start));
    for (int r = 0; r < repeats; ++r)
        saxpy_kernel<<<blocks, threads>>>(a, d_x, d_y, N);
    CUDA_CALL(cudaEventRecord(stop));
    CUDA_CALL(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));
    double secs = ms * 1e-3;
    double flops = 2.0 * N * repeats;
    double bytes = 3.0 * sizeof(float) * N * repeats;
    printf("N=%zu repeats=%d time=%.3f ms GFLOPS=%.3f GB/s=%.3f AI=%.3f\n",
           N, repeats, ms, flops/secs/1e9, bytes/secs/1e9, flops/bytes);

    CUDA_CALL(cudaFree(d_x));
    CUDA_CALL(cudaFree(d_y));
    CUDA_CALL(cudaEventDestroy(start));
    CUDA_CALL(cudaEventDestroy(stop));
    return 0;
}