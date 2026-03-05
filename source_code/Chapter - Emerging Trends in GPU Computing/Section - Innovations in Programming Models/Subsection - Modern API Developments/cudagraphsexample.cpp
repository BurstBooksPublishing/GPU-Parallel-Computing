#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                              \
  cudaError_t err = (call);                                \
  if (err != cudaSuccess) {                                \
    std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
              << " at " << __FILE__ << ":" << __LINE__     \
              << std::endl;                                \
    std::exit(EXIT_FAILURE);                               \
  } } while(0)

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;
  const size_t bytes = n * sizeof(float);
  std::vector<float> hA(n, 1.0f), hB(n, 2.0f), hC(n);

  float *dA, *dB, *dC;
  CUDA_CHECK(cudaMalloc(&dA, bytes));
  CUDA_CHECK(cudaMalloc(&dB, bytes));
  CUDA_CHECK(cudaMalloc(&dC, bytes));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));

  cudaStream_t s;
  CUDA_CHECK(cudaStreamCreate(&s));

  cudaGraph_t graph;
  cudaGraphExec_t graphExec;
  CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  int block = 256;
  int grid = (n + block - 1) / block;
  vecAdd<<<grid, block, 0, s>>>(dA, dB, dC, n);
  CUDA_CHECK(cudaStreamEndCapture(s, &graph));
  CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  const int iterations = 500;
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaGraphLaunch(graphExec, s));
  CUDA_CHECK(cudaStreamSynchronize(s));

  CUDA_CHECK(cudaEventRecord(start, s));
  for (int i = 0; i < iterations; ++i) {
    CUDA_CHECK(cudaGraphLaunch(graphExec, s));
  }
  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float msGraph = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&msGraph, start, stop));

  CUDA_CHECK(cudaEventRecord(start, s));
  for (int i = 0; i < iterations; ++i) {
    vecAdd<<<grid, block, 0, s>>>(dA, dB, dC, n);
  }
  CUDA_CHECK(cudaEventRecord(stop, s));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float msDirect = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&msDirect, start, stop));

  std::cout << "Graph total ms: " << msGraph << ", Direct total ms: " << msDirect << '\n';
  std::cout << "Speedup (direct/graph): " << (msDirect / msGraph) << '\n';

  CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) if (hC[i] != 3.0f) { std::cerr << "Result mismatch\n"; break; }

  CUDA_CHECK(cudaGraphExecDestroy(graphExec));
  CUDA_CHECK(cudaGraphDestroy(graph));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaStreamDestroy(s));
  CUDA_CHECK(cudaFree(dA));
  CUDA_CHECK(cudaFree(dB));
  CUDA_CHECK(cudaFree(dC));
  return 0;
}