#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#define CUDA_CHECK(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { std::cerr << #cmd << " failed: " << cudaGetErrorString(e) << '\n'; std::exit(EXIT_FAILURE); } } while (0)
#define CUBLAS_CHECK(cmd) do { cublasStatus_t s = cmd; if (s != CUBLAS_STATUS_SUCCESS) { std::cerr << #cmd << " failed: " << s << '\n'; std::exit(EXIT_FAILURE); } } while (0)

int main() {
    const int M = 4096, N = 4096, K = 4096;
    const float alpha = 1.0f, beta = 0.0f;

    // Host memory
    std::vector<float> hA(M * K), hB(K * N), hC(M * N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : hA) v = dist(rng);
    for (auto& v : hB) v = dist(rng);

    // Device memory
    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(float) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(float) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(float) * M * N));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(float) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeof(float) * K * N, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // Warm-up
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto t0 = std::chrono::high_resolution_clock::now();
    CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double gflops = (2.0 * M * N * K) / (elapsed * 1e9);
    std::cout << "Elapsed: " << elapsed << " s, GFLOPS: " << gflops << '\n';

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}