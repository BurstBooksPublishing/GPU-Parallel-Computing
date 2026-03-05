#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void chase_kernel(const int* __restrict__ buf, int iterations, unsigned long long* out_cycles) {
    int idx = 0;
    // Warm-up to stabilize caches
    #pragma unroll 16
    for (int i = 0; i < 16; ++i) idx = buf[idx];

    unsigned long long start = clock64();
    #pragma unroll 64
    for (int i = 0; i < iterations; ++i) idx = buf[idx];
    unsigned long long end = clock64();

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out_cycles[0] = end - start;
        // Prevent dead-code elimination
        volatile int sink = idx;
        (void)sink;
    }
}

int main() {
    constexpr size_t N      = 8ULL << 20;   // 8 M integers ≈ 32 MiB
    constexpr int    stride = 16;
    constexpr int    iters  = 1 << 20;

    std::vector<int> host(N);
    for (size_t i = 0; i < N; ++i) host[i] = static_cast<int>((i + stride) % N);

    int* dev_buf    = nullptr;
    unsigned long long* dev_cycles = nullptr;

    cudaMalloc(&dev_buf,    N * sizeof(int));
    cudaMalloc(&dev_cycles, sizeof(unsigned long long));
    cudaMemcpy(dev_buf, host.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    chase_kernel<<<1, 128>>>(dev_buf, iters, dev_cycles);
    cudaDeviceSynchronize();

    unsigned long long cycles = 0;
    cudaMemcpy(&cycles, dev_cycles, sizeof(cycles), cudaMemcpyDeviceToHost);

    printf("Cycles per access: %.3f\n", static_cast<double>(cycles) / iters);

    cudaFree(dev_buf);
    cudaFree(dev_cycles);
    return 0;
}