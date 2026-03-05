#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { std::cerr << #cmd << " failed: " << cudaGetErrorString(e) << '\n'; std::exit(EXIT_FAILURE); } } while (0)
#define CHECK_CUBLAS(cmd) do { cublasStatus_t s = cmd; if (s != CUBLAS_STATUS_SUCCESS) { std::cerr << #cmd << " failed: " << s << '\n'; std::exit(EXIT_FAILURE); } } while (0)

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytesA = size_t(M) * K * sizeof(__half);
    const size_t bytesB = size_t(K) * N * sizeof(__half);
    const size_t bytesC = size_t(M) * N * sizeof(__half);

    CHECK_CUDA(cudaSetDevice(0));

    __half *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytesA));
    CHECK_CUDA(cudaMalloc(&d_B, bytesB));
    CHECK_CUDA(cudaMalloc(&d_C, bytesC));
    CHECK_CUDA(cudaMemset(d_A, 0, bytesA));
    CHECK_CUDA(cudaMemset(d_B, 0, bytesB));
    CHECK_CUDA(cudaMemset(d_C, 0, bytesC));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    const float alpha = 1.0f, beta = 0.0f;
    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              N, M, K,
                              &alpha,
                              d_B, CUDA_R_16F, N,
                              d_A, CUDA_R_16F, K,
                              &beta,
                              d_C, CUDA_R_16F, N,
                              CUDA_R_32F,
                              CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    cudaGraphExec_t exec;
    CHECK_CUDA(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    CHECK_CUDA(cudaStreamSynchronize(stream));
    for (int i = 0; i < 100; ++i) CHECK_CUDA(cudaGraphLaunch(exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    CHECK_CUDA(cudaGraphExecDestroy(exec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}