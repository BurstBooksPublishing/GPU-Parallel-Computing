#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call) do {                                          \
    cudaError_t err = (call);                                     \
    if (err != cudaSuccess) {                                     \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n",            \
                     __FILE__, __LINE__, cudaGetErrorString(err));\
        std::exit(EXIT_FAILURE);                                  \
    }                                                             \
} while (0)

__global__ void scale_kernel(float *data, size_t n, float alpha) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= alpha;
}

int main() {
    const size_t N = 1 << 24;
    const size_t bytes = N * sizeof(float);

    float *A;
    CHECK(cudaSetDevice(0));
    CHECK(cudaMallocManaged(&A, bytes));

    for (size_t i = 0; i < N; ++i) A[i] = 1.0f;

    CHECK(cudaMemAdvise(A, bytes, cudaMemAdviseSetPreferredLocation, 0));

    cudaStream_t s;
    CHECK(cudaStreamCreate(&s));
    CHECK(cudaMemPrefetchAsync(A, bytes, 0, s));

    const int B = 256;
    const int G = (N + B - 1) / B;
    scale_kernel<<<G, B, 0, s>>>(A, N, 2.0f);

    CHECK(cudaStreamSynchronize(s));

    for (size_t i = 0; i < 4; ++i) printf("A[%zu]=%f\n", i, A[i]);

    CHECK(cudaFree(A));
    CHECK(cudaStreamDestroy(s));
    return 0;
}