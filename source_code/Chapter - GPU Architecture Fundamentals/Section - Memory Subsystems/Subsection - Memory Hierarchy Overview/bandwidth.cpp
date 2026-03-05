#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                     \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                         __FILE__, __LINE__, cudaGetErrorString(err));\
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

__global__ void stream_copy(const float* __restrict__ src,
                            float* __restrict__ dst,
                            size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    float acc = 0.0f;

    for (size_t i = idx; i < n; i += stride) {
        float v = src[i];
        dst[i] = v;
        acc += v;
    }

    if (acc != 0.0f && idx == 0)   // only one thread writes
        dst[0] = acc;
}

int main() {
    const size_t N = 1ULL << 26;          // 64M floats
    const size_t bytes = N * sizeof(float);

    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    // fill source to avoid NaN/inf and ensure realistic memory traffic
    CUDA_CHECK(cudaMemset(d_src, 0x3F, bytes)); // ~0.5f pattern

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    stream_copy<<<blocks, threads>>>(d_src, d_dst, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double gbps = (2.0 * bytes) / (ms * 1e6); // read + write
    std::printf("Bytes: %zu, Time: %.3f ms, Bandwidth: %.2f GB/s\n",
                bytes, ms, gbps);

    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return 0;
}