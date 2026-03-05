#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <vector>
#include <iostream>
#include <numeric>
#include <limits>

#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " code=" << err << " \"" << cudaGetErrorString(err) << "\"\n"; \
            std::exit(EXIT_FAILURE);                                      \
        }                                                                 \
    } while (0)

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    std::vector<float> hA(N, 1.0f);
    std::vector<float> hB(N, 2.0f);
    std::vector<float> hC(N);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    nvtxRangePushA("H2D");
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    nvtxRangePop();

    nvtxRangePushA("Kernel");
    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    nvtxRangePop();

    nvtxRangePushA("D2H");
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
    nvtxRangePop();

    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) maxErr = fmax(maxErr, fabs(hC[i] - 3.0f));
    if (maxErr > std::numeric_limits<float>::epsilon() * 3.0f)
        std::cerr << "Validation failed, max error: " << maxErr << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    return 0;
}