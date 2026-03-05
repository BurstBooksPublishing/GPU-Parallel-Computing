#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            std::cerr << "CUDA error " << cudaGetErrorString(e)       \
                      << " at " << __FILE__ << ":" << __LINE__        \
                      << std::endl;                                   \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

#define CUBLAS_CHECK(call)                                            \
    do {                                                              \
        cublasStatus_t s = (call);                                    \
        if (s != CUBLAS_STATUS_SUCCESS) {                             \
            std::cerr << "cuBLAS error " << s                         \
                      << " at " << __FILE__ << ":" << __LINE__        \
                      << std::endl;                                   \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

int main(int argc, char** argv) {
    const int N = (argc > 1) ? std::stoi(argv[1]) : 4096;
    const size_t bytes = static_cast<size_t>(N) * N * sizeof(float);

    std::vector<float> hA(N * static_cast<size_t>(N));
    std::vector<float> hB(N * static_cast<size_t>(N));
    std::vector<float> hC(N * static_cast<size_t>(N), 0.0f);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, N, N,
                             &alpha,
                             dA, N,
                             dB, N,
                             &beta,
                             dC, N));
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    const double secs  = ms * 1e-3;
    const double flops = 2.0 * static_cast<double>(N) * N * N;
    const double tflops = flops / (secs * 1e12);

    std::cout << "N=" << N
              << " Time=" << secs << " s"
              << " Throughput=" << tflops << " TFLOPS\n";

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}