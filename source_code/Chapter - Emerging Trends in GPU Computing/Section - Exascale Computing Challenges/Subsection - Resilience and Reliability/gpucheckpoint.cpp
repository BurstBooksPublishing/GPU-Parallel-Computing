#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

#include <cuda_runtime.h>

static void cudaCheck(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << msg << " -> " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void dummyKernel(float* buf, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] = static_cast<float>(idx);
}

int main() {
    constexpr size_t N = 1ULL << 20;          // 1 M floats
    constexpr int iterations = 10;

    float* d_buf = nullptr;
    cudaCheck(cudaMalloc(&d_buf, N * sizeof(float)), "cudaMalloc");

    dummyKernel<<<(N + 255) / 256, 256>>>(d_buf, N);
    cudaCheck(cudaGetLastError(), "kernel launch");
    cudaCheck(cudaDeviceSynchronize(), "kernel sync");

    float* h_pinned = nullptr;
    cudaCheck(cudaHostAlloc(&h_pinned, N * sizeof(float), cudaHostAllocDefault), "cudaHostAlloc");

    cudaStream_t stream;
    cudaCheck(cudaStreamCreate(&stream), "stream create");

    for (int iter = 0; iter < iterations; ++iter) {
        cudaCheck(cudaMemcpyAsync(h_pinned, d_buf, N * sizeof(float), cudaMemcpyDeviceToHost, stream),
                  "async memcpy");
        cudaCheck(cudaStreamSynchronize(stream), "stream sync");

        std::string tmpName = "checkpoint_" + std::to_string(iter) + ".bin.tmp";
        std::string finalName = "checkpoint_" + std::to_string(iter) + ".bin";

        {
            std::ofstream ofs(tmpName, std::ios::binary);
            if (!ofs) {
                std::cerr << "Failed to open " << tmpName << '\n';
                std::exit(EXIT_FAILURE);
            }
            ofs.write(reinterpret_cast<const char*>(h_pinned), N * sizeof(float));
            if (!ofs) {
                std::cerr << "Failed to write " << tmpName << '\n';
                std::exit(EXIT_FAILURE);
            }
        }                                           // flush & close

        std::rename(tmpName.c_str(), finalName.c_str());

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    cudaCheck(cudaFree(d_buf), "cudaFree");
    cudaCheck(cudaFreeHost(h_pinned), "cudaFreeHost");
    cudaCheck(cudaStreamDestroy(stream), "stream destroy");
    return 0;
}