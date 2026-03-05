#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                    \
    cudaError_t e = (call);                                      \
    if (e != cudaSuccess) {                                      \
        std::cerr << "CUDA error " << cudaGetErrorString(e)      \
                  << " at " << __FILE__ << ":" << __LINE__       \
                  << std::endl;                                  \
        std::exit(EXIT_FAILURE);                                 \
    }                                                            \
} while (0)

int main() {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 2) {
        std::cerr << "Require at least 2 CUDA devices\n";
        return 1;
    }

    int src = 0, dst = 1;
    int canAccess = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccess, dst, src));

    if (canAccess) {
        CUDA_CHECK(cudaSetDevice(dst));
        cudaError_t e = cudaDeviceEnablePeerAccess(src, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);

        CUDA_CHECK(cudaSetDevice(src));
        e = cudaDeviceEnablePeerAccess(dst, 0);
        if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CUDA_CHECK(e);
    }

    const size_t bytes = 64ULL << 20; // 64 MiB
    CUDA_CHECK(cudaSetDevice(src));
    void* d_src = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMemset(d_src, 0xA5, bytes));

    CUDA_CHECK(cudaSetDevice(dst));
    void* d_dst = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, stream));

    if (canAccess) {
        CUDA_CHECK(cudaMemcpyPeerAsync(d_dst, dst, d_src, src, bytes, stream));
    } else {
        void* h_pinned = nullptr;
        CUDA_CHECK(cudaMallocHost(&h_pinned, bytes));

        CUDA_CHECK(cudaSetDevice(src));
        CUDA_CHECK(cudaMemcpyAsync(h_pinned, d_src, bytes, cudaMemcpyDeviceToHost, 0));

        CUDA_CHECK(cudaSetDevice(dst));
        CUDA_CHECK(cudaMemcpyAsync(d_dst, h_pinned, bytes, cudaMemcpyHostToDevice, stream));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFreeHost(h_pinned));
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gb = static_cast<double>(bytes) / (1ULL << 30);
    std::cout << "Transfer size: " << bytes << " bytes, elapsed: " << ms
              << " ms, throughput: " << (gb / (ms * 1e-3)) << " GB/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaSetDevice(src)); CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaSetDevice(dst)); CUDA_CHECK(cudaFree(d_dst));
    return 0;
}