#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = call; if(e != cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

__global__ void warp_compact(const uint32_t *in, uint32_t n, uint32_t *out, uint32_t *out_count) {
  const unsigned full = 0xffffffffu;
  unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= n) return;
  uint32_t val = in[gid];
  int pred = (val % 2 == 0);

  unsigned mask = __ballot_sync(full, pred);
  if (!mask) return;

  unsigned lane = threadIdx.x & 31;
  unsigned rank = __popc(mask & ((1u << lane) - 1u));

  unsigned base;
  if (lane == 0) {
    unsigned warp_total = __popc(mask);
    base = atomicAdd(out_count, warp_total);
  }
  base = __shfl_sync(full, base, 0);

  if (pred) out[base + rank] = val;
}

int main() {
  const uint32_t N = 1u << 20;
  uint32_t *h_in = (uint32_t*)malloc(N * sizeof(uint32_t));
  for (uint32_t i = 0; i < N; ++i) h_in[i] = i;

  uint32_t *d_in, *d_out, *d_count;
  CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_count, sizeof(uint32_t)));
  CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_count, 0, sizeof(uint32_t)));

  cudaEvent_t s, e;
  CUDA_CHECK(cudaEventCreate(&s));
  CUDA_CHECK(cudaEventCreate(&e));

  const int block = 256;
  const int grid = (N + block - 1) / block;
  CUDA_CHECK(cudaEventRecord(s));
  warp_compact<<<grid, block>>>(d_in, N, d_out, d_count);
  CUDA_CHECK(cudaEventRecord(e));
  CUDA_CHECK(cudaEventSynchronize(e));

  float ms;
  CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
  uint32_t out_count;
  CUDA_CHECK(cudaMemcpy(&out_count, d_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  printf("Compacted %u items in %.3f ms (%.2f Mitems/s)\n",
         out_count, ms, out_count / (ms * 1e-3f) / 1e6f);

  CUDA_CHECK(cudaFree(d_in));
  CUDA_CHECK(cudaFree(d_out));
  CUDA_CHECK(cudaFree(d_count));
  free(h_in);
  return 0;
}