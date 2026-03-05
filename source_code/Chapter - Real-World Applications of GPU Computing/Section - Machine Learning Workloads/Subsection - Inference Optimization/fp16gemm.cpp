#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess) {                                                        \
            fprintf(stderr, "%s:%d CUDA error %d (%s)\n",                                \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));                   \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                                               \
    do {                                                                                 \
        cublasStatus_t status = (call);                                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                                           \
            fprintf(stderr, "%s:%d cuBLAS error %d\n", __FILE__, __LINE__, status);      \
            std::exit(EXIT_FAILURE);                                                     \
        }                                                                                \
    } while (0)

int main() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    const int batch = 8;                       // kept for reference; not used in single GEMM
    const size_t bytesA = M * K * sizeof(__half);
    const size_t bytesB = K * N * sizeof(__half);
    const size_t bytesC = M * N * sizeof(__half);

    __half *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost(&h_A, bytesA));
    CUDA_CHECK(cudaMallocHost(&h_B, bytesB));
    CUDA_CHECK(cudaMallocHost(&h_C, bytesC));

    // fill with deterministic pattern
    for (int i = 0; i < M * K; ++i) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; ++i) h_B[i] = __float2half(1.0f);

    __half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, bytesA, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, bytesB, cudaMemcpyHostToDevice, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    CUBLAS_CHECK(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K,
                              &alpha,
                              d_A, CUDA_R_16F, M,
                              d_B, CUDA_R_16F, K,
                              &beta,
                              d_C, CUDA_R_16F, M,
                              CUDA_R_16F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, bytesC, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
    return 0;
}