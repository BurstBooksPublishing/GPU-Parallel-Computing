#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CALL(call) do { cudaError_t e = (call); if(e != cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(EXIT_FAILURE);} } while(0)

__global__ void kernel_local_array(float* __restrict__ out, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= N) return;
  float local[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i) local[i] = tid * 0.001f + i;
  float acc = 0.0f;
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc += local[i] * 0.5f;
  out[tid] = acc;
}

__global__ void kernel_shared_stage(float* __restrict__ out, int N) {
  extern __shared__ float sdata[];
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = threadIdx.x;
  if (tid >= N) return;
  sdata[lane] = tid * 0.001f;
  __syncthreads();
  out[tid] = sdata[lane] * 0.5f;
}

int main() {
  const int N = 1<<20;
  const int threads = 256;
  const int blocks = (N + threads - 1) / threads;
  float *d_out;
  CUDA_CALL(cudaMalloc(&d_out, N * sizeof(float)));
  cudaEvent_t s, e;
  CUDA_CALL(cudaEventCreate(&s));
  CUDA_CALL(cudaEventCreate(&e));

  CUDA_CALL(cudaEventRecord(s));
  kernel_local_array<<<blocks, threads>>>(d_out, N);
  CUDA_CALL(cudaEventRecord(e));
  CUDA_CALL(cudaEventSynchronize(e));
  float ms_local;
  CUDA_CALL(cudaEventElapsedTime(&ms_local, s, e));

  CUDA_CALL(cudaEventRecord(s));
  kernel_shared_stage<<<blocks, threads, threads * sizeof(float)>>>(d_out, N);
  CUDA_CALL(cudaEventRecord(e));
  CUDA_CALL(cudaEventSynchronize(e));
  float ms_shared;
  CUDA_CALL(cudaEventElapsedTime(&ms_shared, s, e));

  printf("local-array kernel: %.3f ms\nshared-stage kernel: %.3f ms\n", ms_local, ms_shared);
  CUDA_CALL(cudaFree(d_out));
  CUDA_CALL(cudaEventDestroy(s));
  CUDA_CALL(cudaEventDestroy(e));
  return EXIT_SUCCESS;
}