#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <cstddef>
#include <cstdio>

__global__ void mem_latency_kernel(const float* __restrict__ data,
                                   float* __restrict__ out,
                                   size_t n,
                                   int iters) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    float acc = 0.0f;
    for (size_t i = tid; i < n; i += stride) {
        float v = data[i];
        #pragma unroll 8
        for (int k = 0; k < iters; ++k)
            acc += v * 0.001f * (k + 1);
        out[i] = acc;
    }
}

int main() {
    const size_t N = 1ULL << 24;
    std::vector<float> h_in(N, 1.0f), h_out(N);

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    int dev;
    cudaGetDevice(&dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    const int target_active_warps = 128;
    int bestBlock = 128, bestActiveWarps = 0;
    for (int block = 32; block <= 1024; block *= 2) {
        int activeBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeBlocksPerSM,
                                                      mem_latency_kernel,
                                                      block,
                                                      0);
        int activeWarps = activeBlocksPerSM * (block / prop.warpSize);
        if (activeWarps >= target_active_warps) {
            bestBlock = block;
            bestActiveWarps = activeWarps;
            break;
        }
        if (activeWarps > bestActiveWarps) {
            bestBlock = block;
            bestActiveWarps = activeWarps;
        }
    }

    const int blocks = (N + bestBlock - 1) / bestBlock;
    cudaDeviceSynchronize();
    const auto t0 = std::chrono::high_resolution_clock::now();
    mem_latency_kernel<<<blocks, bestBlock>>>(d_in, d_out, N, 16);
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(t1 - t0).count();
    const double bytes = double(N) * sizeof(float) * 2.0;

    std::cout << "Device: " << prop.name << '\n'
              << "Block size: " << bestBlock
              << ", active warps/SM: " << bestActiveWarps << '\n'
              << "Throughput: " << (bytes / seconds) / 1e9 << " GB/s\n";

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}