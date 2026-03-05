#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

constexpr int Q = 9;
__constant__ float d_w[Q] = {4.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f, 1.f/9.f,
                               1.f/36.f, 1.f/36.f, 1.f/36.f, 1.f/36.f};
__constant__ int d_ex[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int d_ey[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constexpr float cs2 = 1.f / 3.f;

inline void cudaCheck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); exit(EXIT_FAILURE); }
}

__global__ void lbm_step_pull(const float* __restrict__ f_in, float* __restrict__ f_out,
                              const unsigned char* __restrict__ bc,
                              int nx, int ny, float tau_inv) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ix >= nx || iy >= ny) return;
    int idx = iy * nx + ix;

    float rho = 0.f, ux = 0.f, uy = 0.f;
    float fi[Q];
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        int jx = (ix - d_ex[i] + nx) % nx;
        int jy = (iy - d_ey[i] + ny) % ny;
        fi[i] = f_in[(jy * nx + jx) * Q + i];
        rho += fi[i];
        ux += d_ex[i] * fi[i];
        uy += d_ey[i] * fi[i];
    }
    ux /= rho; uy /= rho;

    float u2 = ux * ux + uy * uy;
    #pragma unroll
    for (int i = 0; i < Q; ++i) {
        float cu = d_ex[i] * ux + d_ey[i] * uy;
        float feq = d_w[i] * rho * (1.f + cu / cs2 + 0.5f * (cu * cu / (cs2 * cs2) - u2 / cs2));
        f_out[idx * Q + i] = bc[idx] ? fi[i] : fi[i] - tau_inv * (fi[i] - feq);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) { fprintf(stderr, "usage: %s nx ny steps\n", argv[0]); return EXIT_FAILURE; }
    int nx = atoi(argv[1]), ny = atoi(argv[2]), steps = atoi(argv[3]);
    size_t cells = size_t(nx) * ny;
    size_t f_bytes = cells * Q * sizeof(float);
    size_t bc_bytes = cells * sizeof(unsigned char);

    float *h_f = (float*)malloc(f_bytes);
    unsigned char *h_bc = (unsigned char*)malloc(bc_bytes);
    for (size_t i = 0; i < cells * Q; ++i) h_f[i] = 4.f / 9.f;
    for (size_t i = 0; i < cells; ++i) h_bc[i] = 0;

    float *d_f0, *d_f1, *d_bc;
    cudaCheck(cudaMalloc(&d_f0, f_bytes), "malloc f0");
    cudaCheck(cudaMalloc(&d_f1, f_bytes), "malloc f1");
    cudaCheck(cudaMalloc(&d_bc, bc_bytes), "malloc bc");
    cudaCheck(cudaMemcpy(d_f0, h_f, f_bytes, cudaMemcpyHostToDevice), "copy f0");
    cudaCheck(cudaMemcpy(d_bc, h_bc, bc_bytes, cudaMemcpyHostToDevice), "copy bc");

    dim3 block(32, 8);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    float tau_inv = 1.6f;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int s = 0; s < steps; ++s) {
        lbm_step_pull<<<grid, block>>>(d_f0, d_f1, d_bc, nx, ny, tau_inv);
        cudaCheck(cudaGetLastError(), "kernel");
        std::swap(d_f0, d_f1);
    }
    cudaCheck(cudaDeviceSynchronize(), "sync");
    auto t1 = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(t1 - t0).count();
    double mlups = double(cells) * steps / 1e6 / secs;
    printf("Domain %dx%d steps %d time %.3fs MLUPS %.2f\n", nx, ny, steps, secs, mlups);

    cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_bc);
    free(h_f); free(h_bc);
    return EXIT_SUCCESS;
}