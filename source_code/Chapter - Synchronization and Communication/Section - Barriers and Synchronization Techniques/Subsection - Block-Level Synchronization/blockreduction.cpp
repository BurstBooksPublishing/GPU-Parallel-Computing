#include <cstdio>
#include <cstdlib>
#include <memory>
#include <numeric>

template <typename T>
__global__ void block_reduce(const T* __restrict__ in, T* __restrict__ out, size_t N) {
    extern __shared__ T sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    T val = (idx < N) ? in[idx] : T(0);
    sdata[tid] = val;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

#define CUDA_CHECK(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

int main() {
    const size_t N = 1 << 20;
    const int TPB = 256;
    const int blocks = (N + TPB - 1) / TPB;

    auto h_in = std::make_unique<float[]>(N);
    std::fill_n(h_in.get(), N, 1.0f);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, blocks * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.get(), N * sizeof(float), cudaMemcpyHostToDevice));

    block_reduce<<<blocks, TPB, TPB * sizeof(float)>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto h_out = std::make_unique<float[]>(blocks);
    CUDA_CHECK(cudaMemcpy(h_out.get(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    double total = std::accumulate(h_out.get(), h_out.get() + blocks, 0.0);

    printf("Total = %.0f (expected %zu)\n", total, N);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}