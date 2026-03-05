#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call)                                                              \
    do {                                                                         \
        cudaError_t e = (call);                                                  \
        if (e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(e));                                      \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

__global__ void sum_loads(const float * __restrict__ data, size_t N,
                          float *out, size_t offset_bytes) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lane = threadIdx.x & 31;
    unsigned warp_id = tid / 32;
    if (warp_id * 32 + lane >= N) return;

    const char *base = reinterpret_cast<const char *>(data) + offset_bytes;
    const float *ptr = reinterpret_cast<const float *>(base);
    size_t idx = warp_id * 32 + lane;

    float s = ptr[idx];   // single coalesced load
    out[tid] = s;         // prevent elimination
}

int main() {
    const size_t N = 1 << 20;
    const size_t threads = 256;
    const size_t blocks = (N + threads - 1) / threads;

    float *d_data = nullptr, *d_out = nullptr;
    CHECK(cudaMalloc(&d_data, (N + 32) * sizeof(float)));
    CHECK(cudaMalloc(&d_out, (N + 32) * sizeof(float)));

    std::vector<float> host(N + 32);
    for (size_t i = 0; i < N + 32; ++i) host[i] = static_cast<float>(i);
    CHECK(cudaMemcpy(d_data, host.data(), (N + 32) * sizeof(float),
                     cudaMemcpyHostToDevice));

    cudaEvent_t s, e;
    CHECK(cudaEventCreate(&s));
    CHECK(cudaEventCreate(&e));
    float ms;

    CHECK(cudaEventRecord(s));
    sum_loads<<<blocks, threads>>>(d_data, N, d_out, 0);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&ms, s, e));
    double bw_aligned = (N * sizeof(float)) / (ms * 1e6);
    printf("Aligned: %.3f ms, throughput %.2f GB/s\n", ms, bw_aligned);

    CHECK(cudaEventRecord(s));
    sum_loads<<<blocks, threads>>>(d_data, N, d_out, 4);
    CHECK(cudaEventRecord(e));
    CHECK(cudaEventSynchronize(e));
    CHECK(cudaEventElapsedTime(&ms, s, e));
    double bw_misaligned = (N * sizeof(float)) / (ms * 1e6);
    printf("Misaligned (+4B): %.3f ms, throughput %.2f GB/s\n", ms,
           bw_misaligned);

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_out));
    CHECK(cudaEventDestroy(s));
    CHECK(cudaEventDestroy(e));
    return 0;
}