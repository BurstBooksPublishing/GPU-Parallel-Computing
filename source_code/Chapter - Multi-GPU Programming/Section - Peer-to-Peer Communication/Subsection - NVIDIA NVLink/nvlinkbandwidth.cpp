#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

static inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    const size_t bytes = (argc > 1) ? std::stoull(argv[1]) : (1ULL << 28); // 256 MB default
    int dev0 = 0, dev1 = 1;
    int ndev;
    check(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
    if (ndev < 2) {
        std::puts("Need at least 2 GPUs");
        return 1;
    }

    int can01 = 0, can10 = 0;
    check(cudaDeviceCanAccessPeer(&can01, dev0, dev1), "cudaDeviceCanAccessPeer 0->1");
    check(cudaDeviceCanAccessPeer(&can10, dev1, dev0), "cudaDeviceCanAccessPeer 1->0");
    std::printf("P2P 0->1: %d, 1->0: %d\n", can01, can10);

    void *d0 = nullptr, *d1 = nullptr;
    check(cudaSetDevice(dev0), "cudaSetDevice 0");
    check(cudaMalloc(&d0, bytes), "cudaMalloc dev0");
    check(cudaMemset(d0, 0xA5, bytes), "cudaMemset dev0");

    check(cudaSetDevice(dev1), "cudaSetDevice 1");
    check(cudaMalloc(&d1, bytes), "cudaMalloc dev1");
    check(cudaMemset(d1, 0x5A, bytes), "cudaMemset dev1");

    if (can01) {
        check(cudaSetDevice(dev0), "cudaSetDevice 0");
        check(cudaDeviceEnablePeerAccess(dev1, 0), "cudaDeviceEnablePeerAccess 0->1");
    }
    if (can10) {
        check(cudaSetDevice(dev1), "cudaSetDevice 1");
        check(cudaDeviceEnablePeerAccess(dev0, 0), "cudaDeviceEnablePeerAccess 1->0");
    }

    cudaStream_t stream;
    check(cudaStreamCreate(&stream), "cudaStreamCreate");

    cudaEvent_t start, stop;
    check(cudaSetDevice(dev0), "cudaSetDevice 0");
    check(cudaEventCreate(&start), "cudaEventCreate start");
    check(cudaEventCreate(&stop), "cudaEventCreate stop");

    const int iterations = 100;
    check(cudaMemcpyPeerAsync(d1, dev1, d0, dev0, bytes, stream), "cudaMemcpyPeerAsync warmup");
    check(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    check(cudaEventRecord(start, stream), "cudaEventRecord start");
    for (int i = 0; i < iterations; ++i) {
        check(cudaMemcpyPeerAsync(d1, dev1, d0, dev0, bytes, stream), "cudaMemcpyPeerAsync");
    }
    check(cudaEventRecord(stop, stream), "cudaEventRecord stop");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    double seconds = ms / 1e3;
    double gb = double(bytes) * iterations / (1024.0 * 1024.0 * 1024.0);
    double bw = gb / seconds;
    std::printf("Transfer %zu bytes x %d iterations: time %.3f ms, bandwidth %.3f GB/s\n",
                bytes, iterations, ms, bw);

    unsigned char hostbuf[4];
    check(cudaMemcpyPeer(d0, dev0, d1, dev1, 4), "cudaMemcpyPeer check");
    check(cudaMemcpy(hostbuf, d0, 4, cudaMemcpyDeviceToHost), "cudaMemcpy to host");
    std::printf("First 4 bytes after round-trip: %02x %02x %02x %02x\n",
                hostbuf[0], hostbuf[1], hostbuf[2], hostbuf[3]);

    check(cudaFree(d0), "cudaFree d0");
    check(cudaFree(d1), "cudaFree d1");
    check(cudaEventDestroy(start), "cudaEventDestroy start");
    check(cudaEventDestroy(stop), "cudaEventDestroy stop");
    check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    return 0;
}