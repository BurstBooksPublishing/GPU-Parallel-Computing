#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) do { cudaError_t e = (call); if (e!=cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1); } } while(0)
#define CHECK_CUBLAS(call) do { cublasStatus_t s = (call); if (s!=CUBLAS_STATUS_SUCCESS) { \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n",__FILE__,__LINE__,s); exit(1); } } while(0)

int main() {
  const int N = 4096;
  const int M = N, K = N;
  const float alpha = 1.0f, beta = 0.0f;
  const size_t elems = static_cast<size_t>(M) * K;

  __half *hA = (__half*)malloc(elems * sizeof(__half));
  __half *hB = (__half*)malloc(elems * sizeof(__half));
  float  *hC = (float*) malloc(elems * sizeof(float));
  for (size_t i = 0; i < elems; ++i) {
    hA[i] = __float2half(static_cast<float>(rand()%3 - 1));
    hB[i] = __float2half(static_cast<float>(rand()%3 - 1));
    hC[i] = 0.0f;
  }

  __half *dA, *dB;
  float  *dC;
  CHECK_CUDA(cudaMalloc(&dA, elems * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dB, elems * sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dC, elems * sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA, hA, elems * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB, elems * sizeof(__half), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC, elems * sizeof(float), cudaMemcpyHostToDevice));

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));
  CHECK_CUDA(cudaEventRecord(start));

  CHECK_CUBLAS(cublasGemmEx(handle,
    CUBLAS_OP_N, CUBLAS_OP_N,
    M, N, K,
    &alpha,
    dA, CUDA_R_16F, M,
    dB, CUDA_R_16F, K,
    &beta,
    dC, CUDA_R_32F, M,
    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

  double gflops = 2.0 * static_cast<double>(M) * N * K / 1e9;
  printf("Size %d, Time %.3f ms, GFLOPS %.2f, TFLOPS %.3f\n",
         N, ms, gflops/(ms/1000.0), gflops/(ms/1000.0)/1000.0);

  cublasDestroy(handle);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  free(hA); free(hB); free(hC);
  return 0;
}