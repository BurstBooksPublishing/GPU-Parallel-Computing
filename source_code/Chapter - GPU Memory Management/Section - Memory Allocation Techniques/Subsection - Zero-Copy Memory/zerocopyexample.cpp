#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CHECK(call)                                                                 \
    do {                                                                            \
        cudaError_t e = (call);                                                     \
        if (e != cudaSuccess) {                                                     \
            std::cerr << "CUDA error " << cudaGetErrorString(e)                     \
                      << " at " << __FILE__ << ":" << __LINE__ << '\n';             \
            std::exit(EXIT_FAILURE);                                                \
        }                                                                           \
    } while (0)

__global__ void sumKernel(const float* __restrict__ in, float* out, std::size_t N) {
    extern __shared__ float smem[];
    std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < N) ? in[idx] : 0.0f;
    smem[threadIdx.x] = val;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) smem[threadIdx.x] += smem[threadIdx.x + offset];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, smem[0]);
}

int main() {
    const std::size_t N = 1ULL << 24;
    const std::size_t bytes = N * sizeof(float);

    float* host_ptr = nullptr;
    CHECK(cudaSetDevice(0));
    CHECK(cudaHostAlloc(&host_ptr, bytes, cudaHostAllocMapped));

    std::fill(host_ptr, host_ptr + N, 1.0f);

    float* device_mapped_ptr = nullptr;
    CHECK(cudaHostGetDevicePointer(&device_mapped_ptr, host_ptr, 0));

    float* d_out = nullptr;
    CHECK(cudaMalloc(&d_out, sizeof(float)));
    CHECK(cudaMemset(d_out, 0, sizeof(float)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    const int smemBytes = threads * sizeof(float);

    CHECK(cudaEventRecord(start));
    sumKernel<<<blocks, threads, smemBytes>>>(device_mapped_ptr, d_out, N);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));

    float result = 0.0f;
    CHECK(cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Sum=" << result << " time(ms)=" << ms << '\n';

    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_out));
    CHECK(cudaFreeHost(host_ptr));
    return 0;
}