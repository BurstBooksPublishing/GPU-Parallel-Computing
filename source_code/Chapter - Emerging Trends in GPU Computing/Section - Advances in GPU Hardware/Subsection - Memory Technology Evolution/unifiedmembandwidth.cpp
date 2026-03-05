#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t e = (call);                                           \
        if (e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                               \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void touch_kernel(float *data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = data[idx] * 1.000001f + 1.0f;
}

int main() {
    const size_t N = 1ULL << 26;                 // 64 M floats
    const size_t bytes = N * sizeof(float);

    // Unified-memory path
    float *umem = nullptr;
    CUDA_CHECK(cudaMallocManaged(&umem, bytes));

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaMemAdvise(umem, bytes, cudaMemAdviseSetPreferredLocation, dev));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s, stream));
    CUDA_CHECK(cudaMemPrefetchAsync(umem, bytes, dev, stream));
    CUDA_CHECK(cudaEventRecord(e, stream));
    CUDA_CHECK(cudaEventSynchronize(e));
    float msPrefetch = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msPrefetch, s, e));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    CUDA_CHECK(cudaEventRecord(s, stream));
    touch_kernel<<<blocks, threads, 0, stream>>>(umem, N);
    CUDA_CHECK(cudaEventRecord(e, stream));
    CUDA_CHECK(cudaEventSynchronize(e));
    float msKernel = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msKernel, s, e));

    printf("Managed prefetch time: %.3f ms, kernel time: %.3f ms\n",
           msPrefetch, msKernel);

    // Pinned-host → device memcpy path
    float *hpin = nullptr;
    CUDA_CHECK(cudaHostAlloc(&hpin, bytes, cudaHostAllocDefault));
    for (size_t i = 0; i < N; ++i) hpin[i] = 1.0f;

    float *dbuf = nullptr;
    CUDA_CHECK(cudaMalloc(&dbuf, bytes));

    CUDA_CHECK(cudaEventRecord(s, stream));
    CUDA_CHECK(cudaMemcpyAsync(dbuf, hpin, bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(e, stream));
    CUDA_CHECK(cudaEventSynchronize(e));
    float msMemcpy = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msMemcpy, s, e));

    printf("Pinned host->device memcpy time: %.3f ms, throughput: %.3f GB/s\n",
           msMemcpy, (bytes / 1e9) / (msMemcpy / 1e3));

    CUDA_CHECK(cudaFree(dbuf));
    CUDA_CHECK(cudaFreeHost(hpin));
    CUDA_CHECK(cudaFree(umem));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}