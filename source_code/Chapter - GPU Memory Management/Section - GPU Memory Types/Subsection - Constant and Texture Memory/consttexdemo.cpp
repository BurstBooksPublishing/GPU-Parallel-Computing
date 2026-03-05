#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <random>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t e = call;                                             \
        if (e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(e));                               \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__constant__ float d_alpha;

__global__ void apply_const_tex(float* __restrict__ out,
                                const float* __restrict__ in,
                                int N,
                                cudaTextureObject_t tex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float alpha = d_alpha;
    float t = tex1Dfetch<float>(tex, i);
    out[i] = alpha * in[i] + t;
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // host memory
    float* h_in  = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);

    // device memory
    float *d_in, *d_out, *d_tex;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_tex, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tex, h_in, bytes, cudaMemcpyHostToDevice));

    // constant
    const float h_alpha = 2.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(d_alpha, &h_alpha, sizeof(float)));

    // texture object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_tex;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.sizeInBytes = bytes;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;

    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr));

    // timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    CUDA_CHECK(cudaEventRecord(start));
    apply_const_tex<<<grid, block>>>(d_out, d_in, N, texObj);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double gb = 3.0 * bytes / 1e9;
    printf("Kernel time: %.3f ms, throughput: %.2f GB/s\n", ms, gb / (ms * 1e-3));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_tex));
    free(h_in);
    free(h_out);
    return 0;
}