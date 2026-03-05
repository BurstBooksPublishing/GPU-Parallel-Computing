#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t e = (expr);                                                          \
        if (e != cudaSuccess) {                                                          \
            std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " at " << __LINE__ \
                      << std::endl;                                                      \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(expr)                                                               \
    do {                                                                                 \
        cublasStatus_t s = (expr);                                                       \
        if (s != CUBLAS_STATUS_SUCCESS) {                                                \
            std::cerr << "cuBLAS Error at " << __LINE__ << std::endl;                    \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

// Convert host FP32 vector to device FP16 array.
static void to_half(const std::vector<float>& in, __half* d_out, size_t n)
{
    std::vector<__half> tmp(n);
    for (size_t i = 0; i < n; ++i)
        tmp[i] = __half(in[i]);
    CUDA_CHECK(cudaMemcpy(d_out, tmp.data(), n * sizeof(__half), cudaMemcpyHostToDevice));
}

int main()
{
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const int m = 1024, n = 1024, k = 1024;
    std::vector<float> A_f(m * k, 1.0f), B_f(k * n, 1.0f);

    __half *dA, *dB;
    float* dC;
    CUDA_CHECK(cudaMalloc(&dA, m * k * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dB, k * n * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&dC, m * n * sizeof(float)));
    CUDA_CHECK(cudaMemset(dC, 0, m * n * sizeof(float)));

    to_half(A_f, dA, m * k);
    to_half(B_f, dB, k * n);

    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              m,
                              n,
                              k,
                              &alpha,
                              dA,
                              CUDA_R_16F,
                              m,
                              dB,
                              CUDA_R_16F,
                              k,
                              &beta,
                              dC,
                              CUDA_R_32F,
                              m,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    float result;
    CUDA_CHECK(cudaMemcpy(&result, dC, sizeof(float), cudaMemcpyDeviceToHost));
    std::cout << "C[0,0]=" << result << std::endl;

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}