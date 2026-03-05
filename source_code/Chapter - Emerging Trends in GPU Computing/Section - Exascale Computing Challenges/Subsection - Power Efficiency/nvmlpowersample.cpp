#include <nvml.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <numeric>
#include <algorithm>

#define NVML_CHECK(call) do { nvmlReturn_t r = (call); if (r != NVML_SUCCESS) { \
    std::cerr << "NVML error: " << nvmlErrorString(r) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); }} while (0)

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::cerr << "CUDA error: " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
    std::exit(EXIT_FAILURE); }} while (0)

__global__ void compute_kernel(float* __restrict__ a,
                               const float* __restrict__ b,
                               const float* __restrict__ c,
                               size_t n, int iters) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = a[idx];
    for (int it = 0; it < iters; ++it)
        val = val * 0.99f + b[idx] * c[idx] * 0.01f;
    a[idx] = val;
}

int main(int argc, char* argv[]) {
    NVML_CHECK(nvmlInit());
    nvmlDevice_t dev;
    NVML_CHECK(nvmlDeviceGetHandleByIndex(0, &dev));

    const size_t n = 1 << 24;               // 16 M elements
    const int iters = 2000;                 // enough work to run ~seconds
    const int block = 256;
    const int grid  = (n + block - 1) / block;

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    // initialise data
    std::vector<float> host(n, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_a, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    std::atomic<bool> running{true};
    std::vector<unsigned int> power_mw_samples;

    // sampling thread
    std::thread sampler([&] {
        while (running) {
            unsigned int mw = 0;
            if (nvmlDeviceGetPowerUsage(dev, &mw) == NVML_SUCCESS)
                power_mw_samples.push_back(mw);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    auto t0 = std::chrono::steady_clock::now();
    compute_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n, iters);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();

    running = false;
    sampler.join();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    if (power_mw_samples.empty()) {
        std::cerr << "No power samples collected\n";
        std::exit(EXIT_FAILURE);
    }
    double avg_mw = std::accumulate(power_mw_samples.begin(), power_mw_samples.end(), 0ULL) /
                    static_cast<double>(power_mw_samples.size());
    double energy_j = avg_mw * 1e-3 * elapsed;

    std::cout << "Elapsed (s): " << elapsed
              << "  Avg Power (W): " << avg_mw * 1e-3
              << "  Energy (J): " << energy_j << std::endl;

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaStreamDestroy(stream));
    NVML_CHECK(nvmlShutdown());
    return 0;
}