#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <thread>
#include <vector>

__global__ void scale_kernel(float* __restrict__ data, size_t n, float scale) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= scale;
}

inline void check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int ngpus = 0;
    check(cudaGetDeviceCount(&ngpus), "cudaGetDeviceCount");
    if (ngpus == 0) {
        std::fprintf(stderr, "No CUDA devices\n");
        return EXIT_FAILURE;
    }

    constexpr size_t N_total = 1ULL << 28;
    constexpr size_t chunk_elems = 1ULL << 20;
    constexpr float scale = 2.0f;

    float* hostBuf = nullptr;
    check(cudaMallocHost(&hostBuf, N_total * sizeof(float)), "cudaMallocHost");
    std::fill(hostBuf, hostBuf + N_total, 1.0f);

    for (int i = 0; i < ngpus; ++i) {
        check(cudaSetDevice(i), "cudaSetDevice");
        for (int j = 0; j < ngpus; ++j) {
            if (i == j) continue;
            int canAccess = 0;
            check(cudaDeviceCanAccessPeer(&canAccess, i, j), "cudaDeviceCanAccessPeer");
            if (canAccess) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e != cudaErrorPeerAccessAlreadyEnabled) check(e, "cudaDeviceEnablePeerAccess");
            }
        }
    }

    std::atomic<size_t> chunkIndex{0};
    const size_t nchunks = (N_total + chunk_elems - 1) / chunk_elems;
    std::vector<std::thread> workers;
    workers.reserve(ngpus);

    for (int dev = 0; dev < ngpus; ++dev) {
        workers.emplace_back([dev, &chunkIndex, nchunks, hostBuf]() {
            check(cudaSetDevice(dev), "worker cudaSetDevice");
            cudaStream_t s;
            check(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreate");

            float* dbuf = nullptr;
            check(cudaMalloc(&dbuf, chunk_elems * sizeof(float)), "cudaMalloc");

            while (true) {
                size_t ci = chunkIndex.fetch_add(1, std::memory_order_relaxed);
                if (ci >= nchunks) break;
                size_t offset = ci * chunk_elems;
                size_t elems = std::min(chunk_elems, N_total - offset);
                size_t bytes = elems * sizeof(float);

                check(cudaMemcpyAsync(dbuf, hostBuf + offset, bytes, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync H2D");
                scale_kernel<<<(elems + 255) / 256, 256, 0, s>>>(dbuf, elems, scale);
                check(cudaGetLastError(), "kernel");
                check(cudaMemcpyAsync(hostBuf + offset, dbuf, bytes, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync D2H");
                check(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            }

            check(cudaFree(dbuf), "cudaFree");
            check(cudaStreamDestroy(s), "cudaStreamDestroy");
        });
    }

    for (auto& t : workers) t.join();

    bool ok = true;
    for (size_t i = 0; i < N_total; ++i) {
        if (hostBuf[i] != scale) { ok = false; break; }
    }
    if (!ok) std::fprintf(stderr, "validation fail\n");

    check(cudaFreeHost(hostBuf), "cudaFreeHost");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}