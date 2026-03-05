#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e, const char* ctx) {
    if (e != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", ctx, cudaGetErrorString(e));
        exit(EXIT_FAILURE);
    }
}

__device__ inline float warpReduceSum(float val) {
    const unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    return val;
}

__global__ void reduceKernel(const float* __restrict__ in, float* out, size_t N) {
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    float v = (gid < N) ? in[gid] : 0.0f;

    float wsum = warpReduceSum(v);
    if ((threadIdx.x & 31) == 0) {
        size_t warpId = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
        out[warpId] = wsum;
    }
}

int main() {
    const size_t N = 1 << 20;
    std::vector<float> h_in(N, 1.0f);

    float *d_in = nullptr, *d_out = nullptr;
    const size_t threads = 256;
    const size_t blocks  = (N + threads - 1) / threads;
    const size_t out_size = blocks * (threads >> 5);

    checkCuda(cudaMalloc(&d_in,  N * sizeof(float)), "cudaMalloc d_in");
    checkCuda(cudaMalloc(&d_out, out_size * sizeof(float)), "cudaMalloc d_out");
    checkCuda(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice), "H2D");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "evt create");
    checkCuda(cudaEventCreate(&stop),  "evt create");
    checkCuda(cudaEventRecord(start),  "evt record");

    reduceKernel<<<blocks, threads>>>(d_in, d_out, N);
    checkCuda(cudaGetLastError(), "kernel");

    checkCuda(cudaEventRecord(stop), "evt record");
    checkCuda(cudaEventSynchronize(stop), "evt sync");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "evt elapsed");

    std::vector<float> h_out(out_size);
    checkCuda(cudaMemcpy(h_out.data(), d_out, out_size * sizeof(float), cudaMemcpyDeviceToHost), "D2H");

    double total = std::accumulate(h_out.begin(), h_out.end(), 0.0);

    printf("N=%zu sum=%.0f time_ms=%.3f warps=%zu\n", N, total, ms, out_size);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}