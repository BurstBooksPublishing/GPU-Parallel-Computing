#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t e = (call);                                                       \
        if (e != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(e));                                           \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

__global__ void naiveStrideKernel(const float *in, float *out, int stride, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    out[gid] = in[gid * stride];  // strided read
}

__global__ void coalescedKernel(const float *in, float *out, int N) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N) return;
    out[gid] = in[gid];  // contiguous read
}

int main(int argc, char **argv) {
    const int N = 1 << 22;  // 4 M elements
    const int TPB = 256;
    int stride = (argc > 1) ? std::atoi(argv[1]) : 1;

    size_t bytes_in = static_cast<size_t>(N) * stride * sizeof(float);
    size_t bytes_out = static_cast<size_t>(N) * sizeof(float);

    float *h_in = nullptr;
    float *h_out = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_in, bytes_in));  // pinned
    CUDA_CHECK(cudaMallocHost(&h_out, bytes_out));

    for (size_t i = 0; i < static_cast<size_t>(N) * stride; ++i) h_in[i] = static_cast<float>(i);

    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_in));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_out));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes_in, cudaMemcpyHostToDevice));

    dim3 blk(TPB);
    dim3 grid((N + TPB - 1) / TPB);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    naiveStrideKernel<<<grid, blk>>>(d_in, d_out, stride, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));

    CUDA_CHECK(cudaEventRecord(start));
    coalescedKernel<<<grid, blk>>>(d_in, d_out, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms_coalesced = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_coalesced, start, stop));

    printf("Stride=%d | naive: %.3f ms | coalesced: %.3f ms | speedup: %.2fx\n",
           stride, ms_naive, ms_coalesced, ms_naive / ms_coalesced);

    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}