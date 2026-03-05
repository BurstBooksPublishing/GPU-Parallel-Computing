#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); } while(0)

__global__ void DataKernel(float *a, size_t N, int work) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < N; i += blockDim.x * gridDim.x) {
        float x = a[i];
        for (int k = 0; k < work; ++k) x = fmaf(1.0000001f, x, 0.000001f);
        a[i] = x;
    }
}

__global__ void TaskKernel(float *out, const int *tasks, int ntasks) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= ntasks) return;
    int work = tasks[tid];
    float x = tid * 1.0f;
    for (int k = 0; k < work; ++k) x = fmaf(0.999999f, x, 0.00001f);
    out[tid] = x;
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double memClockHz = double(prop.memoryClockRate) * 1000.0;
    double B = 2.0 * memClockHz * (prop.memoryBusWidth / 8.0);
    std::cout << "Device bandwidth (theoretical): " << (B / 1e9) << " GB/s\n";

    const size_t N = 10'000'000;
    const int work_per_element = 100;
    const double flops_per_element = 2.0 * work_per_element;
    const double bytes_per_element = sizeof(float);
    const double I = flops_per_element / bytes_per_element;
    const double bound = I * B;
    std::cout << "Estimated intensity I=" << I << " FLOP/byte, I*B=" << (bound/1e12) << " TFLOP/s\n";

    if (bound > 1e12) {
        std::cout << "Choosing data-parallel kernel\n";
        float *d_a;
        CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_a, 0, N * sizeof(float)));
        const int block = 256;
        const int grid = std::min((N + block - 1) / block, 1024);
        DataKernel<<<grid, block>>>(d_a, N, work_per_element);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_a));
    } else {
        std::cout << "Choosing task-parallel kernel\n";
        const int ntasks = 1024;
        std::vector<int> tasks(ntasks);
        std::generate(tasks.begin(), tasks.end(), [n = 0]() mutable { return (n++ % 10) + 10; });
        float *d_out;
        int *d_tasks;
        CUDA_CHECK(cudaMalloc(&d_out, ntasks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_tasks, ntasks * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_tasks, tasks.data(), ntasks * sizeof(int), cudaMemcpyHostToDevice));
        const int block = 128;
        const int grid = (ntasks + block - 1) / block;
        TaskKernel<<<grid, block>>>(d_out, d_tasks, ntasks);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_out));
        CUDA_CHECK(cudaFree(d_tasks));
    }
    return 0;
}