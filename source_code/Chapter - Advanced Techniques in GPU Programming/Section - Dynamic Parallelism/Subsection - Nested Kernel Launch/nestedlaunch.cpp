#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
} while(0)

__global__ void child_kernel(float* data, int offset, int len) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    float v = data[offset + idx];
    #pragma unroll
    for (int i = 0; i < 8; ++i) v = fmaf(v, 1.0001f, 0.0001f * i);
    data[offset + idx] = v;
}

__global__ void parent_kernel(float* data, const int* offsets, const int* lengths, int tasks) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= tasks) return;
    int len = lengths[t];
    if (len <= 0) return;
    child_kernel<<<(len + 127) / 128, 128>>>(data, offsets[t], len);
}

int main() {
    const int N = 1 << 20;
    std::vector<float> h(N);
    for (int i = 0; i < N; ++i) h[i] = static_cast<float>(i);

    float *d;
    CUDA_CHECK(cudaMalloc(&d, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    const int tasks = 1024;
    std::vector<int> h_off(tasks), h_len(tasks);
    int pos = 0;
    for (int i = 0; i < tasks; ++i) {
        int len = 1 + (i & 63);
        if (pos + len > N) len = N - pos;
        h_off[i] = pos;
        h_len[i] = len;
        pos += len;
    }

    int *d_off, *d_len;
    CUDA_CHECK(cudaMalloc(&d_off, tasks * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_len, tasks * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_off, h_off.data(), tasks * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_len, h_len.data(), tasks * sizeof(int), cudaMemcpyHostToDevice));

    parent_kernel<<<(tasks + 255) / 256, 256>>>(d, d_off, d_len, tasks);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h.data(), d, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("sample: %.6f %.6f %.6f\n", h[0], h[1], h[2]);

    CUDA_CHECK(cudaFree(d));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_len));
    return 0;
}