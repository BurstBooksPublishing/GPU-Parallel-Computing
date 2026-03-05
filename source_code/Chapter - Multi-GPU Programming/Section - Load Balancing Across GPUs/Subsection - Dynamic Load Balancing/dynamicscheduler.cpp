#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                               \
    do {                                                                               \
        cudaError_t e = (call);                                                        \
        if (e != cudaSuccess) {                                                        \
            std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __FILE__ \
                      << ":" << __LINE__ << "\n";                                      \
            std::exit(EXIT_FAILURE);                                                   \
        }                                                                              \
    } while (0)

__global__ void saxpy_kernel(const float* x, float* y, float a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    const size_t N            = 200'000'000;
    const size_t base_chunk   = 1'000'000;
    const float  alpha        = 0.3f;
    const float  a            = 2.5f;

    int ngpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ngpus));
    if (ngpus <= 0) {
        std::cerr << "No GPUs found\n";
        return EXIT_FAILURE;
    }

    float *h_x = nullptr, *h_y = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_x, N * sizeof(float), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_y, N * sizeof(float), cudaHostAllocPortable));
    std::memset(h_y, 0, N * sizeof(float));
    std::fill(h_x, h_x + N, 1.0f);

    struct DevState {
        cudaStream_t stream{};
        float *d_x = nullptr, *d_y = nullptr;
        std::atomic<double> throughput{1e6};
        cudaEvent_t start{}, stop{};
    };
    std::vector<DevState> dev(ngpus);

    const size_t max_chunk = base_chunk * 8;
    for (int g = 0; g < ngpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaStreamCreate(&dev[g].stream));
        CUDA_CHECK(cudaMalloc(&dev[g].d_x, max_chunk * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dev[g].d_y, max_chunk * sizeof(float)));
        CUDA_CHECK(cudaEventCreate(&dev[g].start));
        CUDA_CHECK(cudaEventCreate(&dev[g].stop));
    }

    std::atomic<size_t> next_index(0);
    std::mutex throughput_mtx;

    auto worker = [&](int g) {
        CUDA_CHECK(cudaSetDevice(g));
        while (true) {
            double sum_s = 0.0;
            {
                std::lock_guard<std::mutex> lk(throughput_mtx);
                for (int i = 0; i < ngpus; ++i) sum_s += dev[i].throughput.load(std::memory_order_relaxed);
            }
            size_t remaining = N - next_index.load(std::memory_order_relaxed);
            if (remaining == 0) break;

            size_t chunk = base_chunk;
            if (sum_s > 0.0) {
                std::lock_guard<std::mutex> lk(throughput_mtx);
                double s_g = dev[g].throughput.load(std::memory_order_relaxed);
                chunk = static_cast<size_t>(
                    std::max<double>(base_chunk, std::round(remaining * (s_g / sum_s))));
            }
            chunk = std::min({chunk, remaining, max_chunk});

            size_t start = next_index.fetch_add(chunk);
            if (start >= N) break;
            if (start + chunk > N) chunk = N - start;

            CUDA_CHECK(cudaEventRecord(dev[g].start, dev[g].stream));
            CUDA_CHECK(cudaMemcpyAsync(dev[g].d_x, h_x + start, chunk * sizeof(float),
                                       cudaMemcpyHostToDevice, dev[g].stream));
            CUDA_CHECK(cudaMemcpyAsync(dev[g].d_y, h_y + start, chunk * sizeof(float),
                                       cudaMemcpyHostToDevice, dev[g].stream));

            const int block = 256;
            const int grid  = static_cast<int>((chunk + block - 1) / block);
            saxpy_kernel<<<grid, block, 0, dev[g].stream>>>(dev[g].d_x, dev[g].d_y, a, chunk);
            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaMemcpyAsync(h_y + start, dev[g].d_y, chunk * sizeof(float),
                                       cudaMemcpyDeviceToHost, dev[g].stream));
            CUDA_CHECK(cudaEventRecord(dev[g].stop, dev[g].stream));
            CUDA_CHECK(cudaEventSynchronize(dev[g].stop));

            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, dev[g].start, dev[g].stop));
            double seconds = ms * 1e-3;
            if (seconds > 0.0) {
                double sample_throughput = double(chunk) / seconds;
                double old = dev[g].throughput.load(std::memory_order_relaxed);
                dev[g].throughput.store(alpha * sample_throughput + (1.0 - alpha) * old,
                                        std::memory_order_relaxed);
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(ngpus);
    for (int g = 0; g < ngpus; ++g) threads.emplace_back(worker, g);
    for (auto& t : threads) t.join();

    for (int g = 0; g < ngpus; ++g) {
        CUDA_CHECK(cudaSetDevice(g));
        CUDA_CHECK(cudaFree(dev[g].d_x));
        CUDA_CHECK(cudaFree(dev[g].d_y));
        CUDA_CHECK(cudaStreamDestroy(dev[g].stream));
        CUDA_CHECK(cudaEventDestroy(dev[g].start));
        CUDA_CHECK(cudaEventDestroy(dev[g].stop));
    }
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));
    std::cout << "Done\n";
    return EXIT_SUCCESS;
}