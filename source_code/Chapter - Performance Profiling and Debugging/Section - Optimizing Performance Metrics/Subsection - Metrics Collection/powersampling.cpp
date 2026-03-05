#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <nvml.h>
#include <thread>
#include <vector>

__global__ void vecAdd(const float* A, const float* B, float* C, size_t N) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t e = (call);                                           \
        if (e != cudaSuccess) {                                           \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) << '\n'; \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

#define NVML_CHECK(call)                                                  \
    do {                                                                  \
        nvmlReturn_t r = (call);                                          \
        if (r != NVML_SUCCESS) {                                          \
            std::cerr << "NVML Error: " << nvmlErrorString(r) << '\n';    \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

int main() {
    constexpr size_t N = 1ULL << 22;
    constexpr size_t bytes = N * sizeof(float);
    std::vector<float> hA(N, 1.0f), hB(N, 2.0f), hC(N);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

    NVML_CHECK(nvmlInit());
    nvmlDevice_t nvmlDevice;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &nvmlDevice));

    std::atomic<bool> sampling{true};
    std::vector<double> power_samples_w;
    power_samples_w.reserve(10000);  // pre-allocate to avoid realloc

    std::thread sampler([&]() {
        while (sampling.load(std::memory_order_acquire)) {
            unsigned int mw = 0;
            if (nvmlDeviceGetPowerUsage(nvmlDevice, &mw) == NVML_SUCCESS)
                power_samples_w.push_back(mw / 1000.0);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    CUDA_CHECK(cudaEventRecord(start));
    vecAdd<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    sampling.store(false, std::memory_order_release);
    sampler.join();

    double avg_power_w = 0.0;
    if (!power_samples_w.empty())
        avg_power_w = std::accumulate(power_samples_w.begin(), power_samples_w.end(), 0.0) /
                      power_samples_w.size();

    double time_s = ms / 1000.0;
    double energy_j = avg_power_w * time_s;

    std::cout << std::fixed << std::setprecision(6)
              << "Kernel time (s): " << time_s << '\n'
              << "Average power (W): " << avg_power_w << '\n'
              << "Estimated energy (J): " << energy_j << '\n';

    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    NVML_CHECK(nvmlShutdown());
    return 0;
}