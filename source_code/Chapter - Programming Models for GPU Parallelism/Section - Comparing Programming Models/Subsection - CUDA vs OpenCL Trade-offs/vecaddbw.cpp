#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#define CHECK(call) do {                               \
  cudaError_t e = (call);                              \
  if (e != cudaSuccess) {                              \
    fprintf(stderr, "CUDA error %s:%d: %s\n",          \
            __FILE__, __LINE__, cudaGetErrorString(e));\
    exit(EXIT_FAILURE);                                \
  } } while(0)

__global__ void vec_add(const float* a, const float* b, float* c, size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
  size_t N = (argc > 1) ? std::strtoull(argv[1], nullptr, 10) : 1ULL << 26;
  size_t bytes = N * sizeof(float);

  float *hA = nullptr, *hB = nullptr, *hC = nullptr;
  CHECK(cudaMallocHost(&hA, bytes));
  CHECK(cudaMallocHost(&hB, bytes));
  CHECK(cudaMallocHost(&hC, bytes));

  for (size_t i = 0; i < N; ++i) {
    hA[i] = 1.0f;
    hB[i] = 2.0f;
  }

  float *dA = nullptr, *dB = nullptr, *dC = nullptr;
  CHECK(cudaMalloc(&dA, bytes));
  CHECK(cudaMalloc(&dB, bytes));
  CHECK(cudaMalloc(&dC, bytes));

  auto t0 = std::chrono::steady_clock::now();
  CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
  CHECK(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  double t_h2d = std::chrono::duration<double>(t1 - t0).count();

  const int block = 256;
  int grid = static_cast<int>((N + block - 1) / block);
  t0 = std::chrono::steady_clock::now();
  vec_add<<<grid, block>>>(dA, dB, dC, N);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  t1 = std::chrono::steady_clock::now();
  double t_kernel = std::chrono::duration<double>(t1 - t0).count();

  double bw_kernel = (3.0 * bytes) / (t_kernel * 1e9);
  double bw_h2d = (2.0 * bytes) / (t_h2d * 1e9);

  printf("N=%zu, kernel_time=%.6f s, kernel_BW=%.2f GB/s\n", N, t_kernel, bw_kernel);
  printf("host_to_device_time=%.6f s, host_to_device_BW=%.2f GB/s\n", t_h2d, bw_h2d);

  CHECK(cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost));

  bool ok = true;
  for (size_t i = 0; i < N; ++i) ok &= (hC[i] == 3.0f);
  printf("validation: %s\n", ok ? "PASS" : "FAIL");

  CHECK(cudaFree(dA));
  CHECK(cudaFree(dB));
  CHECK(cudaFree(dC));
  CHECK(cudaFreeHost(hA));
  CHECK(cudaFreeHost(hB));
  CHECK(cudaFreeHost(hC));
  return 0;
}