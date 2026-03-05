#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t e = (call);                                           \
    if (e != cudaSuccess) {                                           \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
              cudaGetErrorString(e));                                 \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

__device__ int d_barrier_cnt;
__device__ int d_barrier_epoch;

__device__ void software_grid_barrier(int numBlocks, int epoch) {
  if (threadIdx.x == 0) {                       // single thread per block
    int ticket = atomicAdd(&d_barrier_cnt, 1);
    __threadfence_system();
    if (ticket == numBlocks - 1) {              // last block
      d_barrier_cnt = 0;
      __threadfence_system();
      d_barrier_epoch = epoch + 1;
    }
  }
  __syncthreads();                              // wait for block leader
  while (d_barrier_epoch < epoch + 1) {         // spin
    __threadfence_system();
  }
}

__global__ void cooperative_kernel(float* data, int N, int num_phases) {
  grid_group grid = this_grid();
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int phase = 0; phase < num_phases; ++phase) {
    if (gid < N) data[gid] += 1.0f;
    grid.sync();
  }
}

__global__ void persistent_kernel(float* data, int N, int num_phases, int numBlocks) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int phase = 0; phase < num_phases; ++phase) {
    if (gid < N) data[gid] += 1.0f;
    software_grid_barrier(numBlocks, phase);
  }
}

int main() {
  const int N = 1 << 20;
  const int threadsPerBlock = 256;
  const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  const int num_phases = 10;

  float* d_data;
  CUDA_CHECK(cudaMalloc(&d_data, N * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_data, 0, N * sizeof(float)));

  int coopAttr = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&coopAttr, cudaDevAttrCooperativeLaunch, 0));

  if (coopAttr) {
    void* args[] = {&d_data, (void*)&N, (void*)&num_phases};
    dim3 grid(numBlocks);
    dim3 block(threadsPerBlock);
    CUDA_CHECK(cudaLaunchCooperativeKernel((void*)cooperative_kernel, grid, block, args));
  } else {
    int zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(d_barrier_cnt, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_barrier_epoch, &zero, sizeof(int)));
    persistent_kernel<<<numBlocks, threadsPerBlock>>>(d_data, N, num_phases, numBlocks);
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaFree(d_data));
  return 0;
}