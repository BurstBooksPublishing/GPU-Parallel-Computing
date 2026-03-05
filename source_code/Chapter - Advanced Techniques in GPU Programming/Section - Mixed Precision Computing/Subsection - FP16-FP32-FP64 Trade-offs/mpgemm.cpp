#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                          __FILE__, __LINE__, cudaGetErrorString(e)); \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

#define CHECK_CUBLAS(call)                                            \
    do {                                                              \
        cublasStatus_t s = (call);                                    \
        if (s != CUBLAS_STATUS_SUCCESS) {                             \
            std::fprintf(stderr, "cuBLAS error %s:%d: %d\n",          \
                          __FILE__, __LINE__, static_cast<int>(s));   \
            std::exit(EXIT_FAILURE);                                  \
        }                                                             \
    } while (0)

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const float alpha = 1.0f, beta = 0.0f;

    const size_t bytesA = M * K * sizeof(__half);
    const size_t bytesB = K * N * sizeof(__half);
    const size_t bytesC = M * N * sizeof(float);

    __half *hA = static_cast<__half*>(std::malloc(bytesA));
    __half *hB = static_cast<__half*>(std::malloc(bytesB));
    float  *hC = static_cast<float*>(std::malloc(bytesC));

    for (int i = 0; i < M * K; ++i) hA[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) hB[i] = __float2half(1.0f);
    for (int i = 0; i < M * N; ++i) hC[i] = 0.0f;

    __half *dA, *dB;
    float  *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));

    CHECK_CUDA(cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dC, hC, bytesC, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              dB, CUDA_R_16F, N,
                              dA, CUDA_R_16F, K,
                              &beta,
                              dC, CUDA_R_32F, N,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUDA(cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost));
    std::printf("C[0]=%f\n", hC[0]);

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    std::free(hA);
    std::free(hB);
    std::free(hC);
    return 0;
}