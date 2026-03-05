#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA:%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void uniform_kernel(const float *a, float *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = a[idx];
    #pragma unroll
    for (int i = 0; i < 16; ++i) v = fmaf(v, 1.000001f, 0.000001f);
    b[idx] = v;
}

__global__ void divergent_kernel(const float *a, float *b, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    bool take = (threadIdx.x & 31) < 16; // warp-divergent predicate
    float v = a[idx];
    #pragma unroll
    for (int i = 0; i < 16; ++i) v = take ? fmaf(v, 1.000001f, 0.000001f)
                                           : fmaf(v, 0.999999f, 0.000002f);
    b[idx] = v;
}

int main() {
    const size_t N = 1ULL << 24;
    const size_t bytes = N * sizeof(float);
    float *h_a, *h_b;
    CHECK(cudaMallocHost(&h_a, bytes));
    CHECK(cudaMallocHost(&h_b, bytes));
    for (size_t i = 0; i < N; ++i) h_a[i] = static_cast<float>(i);

    float *d_a, *d_b;
    CHECK(cudaMalloc(&d_a, bytes));
    CHECK(cudaMalloc(&d_b, bytes));
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    uniform_kernel<<<blocks, threads>>>(d_a, d_b, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    double bw = (2.0 * bytes) / (ms * 1e6);
    printf("Uniform kernel:   %.3f ms  %.2f GB/s\n", ms, bw);

    CHECK(cudaEventRecord(start));
    divergent_kernel<<<blocks, threads>>>(d_a, d_b, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    bw = (2.0 * bytes) / (ms * 1e6);
    printf("Divergent kernel: %.3f ms  %.2f GB/s\n", ms, bw);

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFreeHost(h_a));
    CHECK(cudaFreeHost(h_b));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    return 0;
}