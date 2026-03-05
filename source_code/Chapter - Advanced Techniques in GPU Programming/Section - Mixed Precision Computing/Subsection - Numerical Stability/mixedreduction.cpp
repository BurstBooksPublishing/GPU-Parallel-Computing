#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

static inline void check(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s\n", cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void reduce_half_to_float(const __half *in, size_t N, float *out) {
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float sum = 0.0f, c = 0.0f;                 // Kahan accumulator & compensation
    for (size_t i = tid; i < N; i += stride) {
        float y = __half2float(in[i]) - c;
        float t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (threadIdx.x % warpSize == 0) atomicAdd(out, sum);
}

int main() {
    const size_t N = 1ULL << 24;
    __half *d_in;
    float  *d_out;
    check(cudaMalloc(&d_in,  N * sizeof(__half)));
    check(cudaMalloc(&d_out, sizeof(float)));
    check(cudaMemset(d_out, 0, sizeof(float)));

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    reduce_half_to_float<<<blocks, threads>>>(d_in, N, d_out);
    check(cudaDeviceSynchronize());

    float host_out;
    check(cudaMemcpy(&host_out, d_out, sizeof(float), cudaMemcpyDeviceToHost));
    std::printf("reduced sum = %g\n", host_out);

    check(cudaFree(d_in));
    check(cudaFree(d_out));
    return 0;
}