#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

static inline void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); exit(EXIT_FAILURE); }
}

struct Body { float x, y, z, vx, vy, vz, m; };

__global__ void nbody_tile_kernel(const Body* __restrict__ sources,
                                  Body* __restrict__ targets,
                                  int N, float dt, float eps2) {
    extern __shared__ float sh[];
    float* sx = sh;
    float* sy = sh + blockDim.x;
    float* sz = sh + 2 * blockDim.x;
    float* sm = sh + 3 * blockDim.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;

    Body bi = targets[tid];
    float tx = bi.x, ty = bi.y, tz = bi.z;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    for (int tile = 0; tile * blockDim.x < N; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < N) {
            Body b = sources[idx];
            sx[threadIdx.x] = b.x;
            sy[threadIdx.x] = b.y;
            sz[threadIdx.x] = b.z;
            sm[threadIdx.x] = b.m;
        } else {
            sx[threadIdx.x] = 0.0f;
            sy[threadIdx.x] = 0.0f;
            sz[threadIdx.x] = 0.0f;
            sm[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        #pragma unroll 8
        for (int j = 0; j < blockDim.x; ++j) {
            float dx = sx[j] - tx;
            float dy = sy[j] - ty;
            float dz = sz[j] - tz;
            float dist2 = dx * dx + dy * dy + dz * dz + eps2;
            float invDist = rsqrtf(dist2);
            float invDist3 = invDist * invDist * invDist;
            float f = sm[j] * invDist3;
            ax += f * dx;
            ay += f * dy;
            az += f * dz;
        }
        __syncthreads();
    }

    bi.vx += ax * dt;
    bi.vy += ay * dt;
    bi.vz += az * dt;
    bi.x += bi.vx * dt;
    bi.y += bi.vy * dt;
    bi.z += bi.vz * dt;
    targets[tid] = bi;
}

int main() {
    const int N = 20000;
    const int steps = 10;
    const float dt = 0.01f;
    const float eps2 = 0.01f * 0.01f;
    const int blockSize = 256;

    std::vector<Body> host(N);
    for (int i = 0; i < N; ++i) {
        host[i] = { static_cast<float>(i), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
    }

    Body *d_src, *d_tgt;
    size_t bytes = N * sizeof(Body);
    cudaCheck(cudaMalloc(&d_src, bytes), "alloc src");
    cudaCheck(cudaMalloc(&d_tgt, bytes), "alloc tgt");
    cudaCheck(cudaMemcpy(d_src, host.data(), bytes, cudaMemcpyHostToDevice), "copy H2D");

    dim3 grid((N + blockSize - 1) / blockSize);
    size_t shmem = 4 * blockSize * sizeof(float);

    for (int s = 0; s < steps; ++s) {
        nbody_tile_kernel<<<grid, blockSize, shmem>>>(d_src, d_tgt, N, dt, eps2);
        cudaCheck(cudaPeekAtLastError(), "kernel launch");
        cudaCheck(cudaDeviceSynchronize(), "kernel sync");
        std::swap(d_src, d_tgt);
    }

    cudaCheck(cudaMemcpy(host.data(), d_src, bytes, cudaMemcpyDeviceToHost), "copy D2H");
    cudaCheck(cudaFree(d_src), "free src");
    cudaCheck(cudaFree(d_tgt), "free tgt");
    printf("done\n");
    return 0;
}