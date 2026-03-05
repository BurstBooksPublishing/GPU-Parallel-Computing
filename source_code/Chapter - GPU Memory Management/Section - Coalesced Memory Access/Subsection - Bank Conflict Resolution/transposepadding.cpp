#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>

#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(EXIT_FAILURE); } } while(0)

constexpr int TILE = 32;

__global__ void transpose_pad(const float* __restrict__ src, float* __restrict__ dst,
                              int width, int height) {
    __shared__ float tile[TILE][TILE+1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = src[y * width + x];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    __syncthreads();

    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;

    if (tx < height && ty < width)
        dst[tx * height + ty] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int width  = 4096;
    const int height = 4096;
    const size_t bytes = size_t(width) * height * sizeof(float);

    float *h_in  = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    if (!h_in || !h_out) { fprintf(stderr, "Host alloc failed\n"); return 1; }

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < size_t(width) * height; ++i) h_in[i] = dist(rng);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
    transpose_pad<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int r = 0; r < height && ok; ++r)
        for (int c = 0; c < width && ok; ++c)
            if (h_out[r * height + c] != h_in[c * width + r]) ok = false;

    printf("%s\n", ok ? "Transpose successful." : "FAILURE");
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);
    return ok ? 0 : 1;
}