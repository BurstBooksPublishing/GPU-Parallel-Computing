#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                         \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,          \
              cudaGetErrorString(err));                                      \
      std::exit(EXIT_FAILURE);                                               \
    }                                                                        \
  } while (0)

__global__ void accumulate_kernel(const int * __restrict__ data,
                                  int n,
                                  unsigned long long *global_sum)
{
    extern __shared__ unsigned long long sdata[];

    unsigned long long local = 0ULL;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = tid; i < n; i += stride)
        local += static_cast<unsigned long long>(data[i]);

    sdata[threadIdx.x] = local;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) sdata[threadIdx.x] += sdata[threadIdx.x + offset];
        __syncthreads();
    }

    if (threadIdx.x == 0) atomicAdd(global_sum, sdata[0]);
}

int main() {
    const int N = 1 << 24;
    const int block = 256;
    const int maxGrid = 65535;
    int grid = (N + block - 1) / block;
    if (grid > maxGrid) grid = maxGrid;

    size_t bytes = N * sizeof(int);
    int *h_data = nullptr;
    CHECK(cudaMallocHost(&h_data, bytes));          // pinned host memory
    for (int i = 0; i < N; ++i) h_data[i] = 1;

    int *d_data = nullptr;
    unsigned long long *d_sum = nullptr;
    CHECK(cudaMalloc(&d_data, bytes));
    CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));

    CHECK(cudaMemcpyAsync(d_data, h_data, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemsetAsync(d_sum, 0, sizeof(unsigned long long)));

    accumulate_kernel<<<grid, block, block * sizeof(unsigned long long)>>>(d_data, N, d_sum);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    unsigned long long h_sum = 0ULL;
    CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    printf("Sum = %llu (expected %d)\n", h_sum, N);

    CHECK(cudaFree(d_data));
    CHECK(cudaFree(d_sum));
    CHECK(cudaFreeHost(h_data));
    return 0;
}