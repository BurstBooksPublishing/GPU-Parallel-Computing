#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                      \
    cudaError_t e = (call);                                        \
    if (e != cudaSuccess) {                                        \
        std::fprintf(stderr, "CUDA ERROR %s:%d: %s\n",             \
                     __FILE__, __LINE__, cudaGetErrorString(e));   \
        std::exit(EXIT_FAILURE);                                   \
    } } while (0)

int main() {
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount < 2) {
        std::fprintf(stderr, "Need at least two GPUs\n");
        return EXIT_FAILURE;
    }

    const int devA = 0, devB = 1;

    int canAB = 0, canBA = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAB, devA, devB));
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canBA, devB, devA));

    if (canAB) {
        CUDA_CHECK(cudaSetDevice(devA));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(devB, 0));
    }
    if (canBA) {
        CUDA_CHECK(cudaSetDevice(devB));
        CUDA_CHECK(cudaDeviceEnablePeerAccess(devA, 0));
    }

    const size_t bytes = 256ULL * 1024 * 1024;

    CUDA_CHECK(cudaSetDevice(devA));
    void* dA = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));

    CUDA_CHECK(cudaSetDevice(devB));
    void* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, bytes));

    cudaStream_t sA = nullptr, sB = nullptr;
    CUDA_CHECK(cudaSetDevice(devA));
    CUDA_CHECK(cudaStreamCreate(&sA));
    CUDA_CHECK(cudaSetDevice(devB));
    CUDA_CHECK(cudaStreamCreate(&sB));

    if (canAB) {
        CUDA_CHECK(cudaSetDevice(devA));
        CUDA_CHECK(cudaMemcpyPeerAsync(dB, devB, dA, devA, bytes, sA));
        CUDA_CHECK(cudaStreamSynchronize(sA));
    } else {
        void* h = std::malloc(bytes);
        CUDA_CHECK(cudaSetDevice(devA));
        CUDA_CHECK(cudaMemcpy(h, dA, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaSetDevice(devB));
        CUDA_CHECK(cudaMemcpy(dB, h, bytes, cudaMemcpyHostToDevice));
        std::free(h);
    }

    CUDA_CHECK(cudaSetDevice(devA));
    cudaFree(dA);
    cudaStreamDestroy(sA);
    CUDA_CHECK(cudaSetDevice(devB));
    cudaFree(dB);
    cudaStreamDestroy(sB);

    return 0;
}