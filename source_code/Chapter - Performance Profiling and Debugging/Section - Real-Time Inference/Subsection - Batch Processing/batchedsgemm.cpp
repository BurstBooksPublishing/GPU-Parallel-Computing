#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error %s at %d\n",             \
                         cudaGetErrorString(e), __LINE__);            \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

#define CHECK_CUBLAS(call)                                            \
    do {                                                              \
        cublasStatus_t s = (call);                                    \
        if (s != CUBLAS_STATUS_SUCCESS) {                             \
            std::fprintf(stderr, "cuBLAS error %d at %d\n",           \
                         static_cast<int>(s), __LINE__);              \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

int main() {
    const int M = 512, N = 512, K = 1024;
    const int batch = 32;

    const size_t bytesA = size_t(M) * K * sizeof(float);
    const size_t bytesB = size_t(K) * N * batch * sizeof(float);
    const size_t bytesC = size_t(M) * N * batch * sizeof(float);

    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    CHECK_CUDA(cudaHostAlloc(&hA, bytesA, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hB, bytesB, cudaHostAllocDefault));
    CHECK_CUDA(cudaHostAlloc(&hC, bytesC, cudaHostAllocDefault));

    std::fill(hA, hA + M * K, 1.0f);
    std::fill(hB, hB + K * N * batch, 1.0f);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    CHECK_CUDA(cudaMemcpyAsync(dA, hA, bytesA, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(dB, hB, bytesB, cudaMemcpyHostToDevice, stream));

    const float alpha = 1.0f, beta = 0.0f;
    const long long strideA = 0;
    const long long strideB = long long(K) * N;
    const long long strideC = long long(M) * N;

    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        dA, M, strideA,
        dB, K, strideB,
        &beta,
        dC, M, strideC,
        batch));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    CHECK_CUBLAS(cublasSgemmStridedBatched(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        dA, M, strideA,
        dB, K, strideB,
        &beta,
        dC, M, strideC,
        batch));
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double throughput = batch / (ms * 1e-3);

    std::printf("Batch=%d  Latency=%.3f ms  Throughput=%.1f samples/s\n",
                batch, ms, throughput);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    CHECK_CUDA(cudaFreeHost(hA));
    CHECK_CUDA(cudaFreeHost(hB));
    CHECK_CUDA(cudaFreeHost(hC));
    return 0;
}