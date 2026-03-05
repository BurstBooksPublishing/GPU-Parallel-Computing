#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t e = (call);                                       \
        if (e != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(e));       \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

__global__ void fma_kernel(const float *a, const float *b, float *c, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float x = a[i];
    float y = b[i];
    float acc = 0.0f;
    #pragma unroll 8
    for (int k = 0; k < K; ++k) acc = fmaf(x, y, acc);
    c[i] = acc;
}

int main(int argc, char **argv) {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("Running on GPU: %s\n", prop.name);

    const int N = (argc > 1) ? std::atoi(argv[1]) : 1 << 22;
    const int K = (argc > 2) ? std::atoi(argv[2]) : 16;

    size_t bytes = N * sizeof(float);
    float *hA, *hB, *hC;
    CUDA_CHECK(cudaMallocHost(&hA, bytes));
    CUDA_CHECK(cudaMallocHost(&hB, bytes));
    CUDA_CHECK(cudaMallocHost(&hC, bytes));
    for (int i = 0; i < N; ++i) { hA[i] = 1.0f + i * 1e-6f; hB[i] = 0.5f; }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    fma_kernel<<<grid, block>>>(dA, dB, dC, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double seconds = ms * 1e-3;

    double flops = double(N) * K * 2.0;
    double bytesMoved = double(N) * 3 * sizeof(float);
    double gflops = flops / 1e9 / seconds;
    double bandwidthGBs = bytesMoved / 1e9 / seconds;

    printf("N=%d K=%d time=%.3f ms GFLOPS=%.2f GB/s=%.2f\n", N, K, ms, gflops, bandwidthGBs);

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFreeHost(hA));
    CUDA_CHECK(cudaFreeHost(hB));
    CUDA_CHECK(cudaFreeHost(hC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}