#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <algorithm>

#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t e = (call);                                                        \
        if (e != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(e));                                            \
            cudaDeviceReset();                                                         \
            exit(EXIT_FAILURE);                                                        \
        }                                                                              \
    } while (0)

__global__ void read_kernel(const float * __restrict__ data,
                            size_t stride,
                            size_t iters,
                            float * __restrict__ sink) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx = tid * stride;
    float acc = 0.0f;
    #pragma unroll 8
    for (size_t i = 0; i < iters; ++i) {
        acc += data[idx];
        idx = (idx + 1) % (gridDim.x * blockDim.x * stride);
    }
    sink[tid] = acc;
}

int main() {
    const int threads_per_block = 256;
    const int blocks = 128;
    const size_t num_threads = static_cast<size_t>(threads_per_block) * blocks;
    const size_t stride = 1;
    const size_t iters = 1024;
    const size_t elems = num_threads * stride + 128;

    float *d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, elems * sizeof(float)));
    std::vector<float> h_buf(elems, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_buf, h_buf.data(), elems * sizeof(float), cudaMemcpyHostToDevice));

    float *d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sink, num_threads * sizeof(float)));

    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));

    CUDA_CHECK(cudaEventRecord(s));
    read_kernel<<<blocks, threads_per_block>>>(d_buf, stride, iters, d_sink);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms_aligned;
    CUDA_CHECK(cudaEventElapsedTime(&ms_aligned, s, e));

    float *d_buf_misaligned = reinterpret_cast<float*>(reinterpret_cast<char*>(d_buf) + sizeof(float));
    CUDA_CHECK(cudaEventRecord(s));
    read_kernel<<<blocks, threads_per_block>>>(d_buf_misaligned, stride, iters, d_sink);
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms_misaligned;
    CUDA_CHECK(cudaEventElapsedTime(&ms_misaligned, s, e));

    const double bytes_loaded = static_cast<double>(num_threads) * iters * sizeof(float);
    const double bw_aligned = bytes_loaded / (ms_aligned * 1e-3) / 1e9;
    const double bw_misaligned = bytes_loaded / (ms_misaligned * 1e-3) / 1e9;

    printf("Aligned:    %7.2f ms  %.2f GB/s\n", ms_aligned, bw_aligned);
    printf("Misaligned: %7.2f ms  %.2f GB/s\n", ms_misaligned, bw_misaligned);

    CUDA_CHECK(cudaFree(d_buf));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return 0;
}