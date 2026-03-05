#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) { 
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e)); 
        std::exit(EXIT_FAILURE); 
    }
}

__device__ void process_item(int idx, const int *offsets, const int *adj, int *out) {
    int start = offsets[idx], end = offsets[idx+1];
    int acc = 0;
    for (int i = start; i < end; ++i) {
        acc += adj[i];
    }
    out[idx] = acc;
}

__global__ void persistent_work_queue(const int *offsets, const int *adj,
                                      int *out, int n_items, int *global_idx) {
    while (true) {
        int idx = atomicAdd(global_idx, 1);
        if (idx >= n_items) break;
        process_item(idx, offsets, adj, out);
    }
}

int main() {
    const int n_items = 1 << 20;
    int *d_offsets, *d_adj, *d_out, *d_global_idx;

    checkCuda(cudaMalloc(&d_offsets, (n_items + 1) * sizeof(int)), "cudaMalloc offsets");
    checkCuda(cudaMalloc(&d_adj, 10 * n_items * sizeof(int)), "cudaMalloc adj");
    checkCuda(cudaMalloc(&d_out, n_items * sizeof(int)), "cudaMalloc out");
    checkCuda(cudaMalloc(&d_global_idx, sizeof(int)), "cudaMalloc global_idx");
    checkCuda(cudaMemset(d_global_idx, 0, sizeof(int)), "memset global_idx");

    persistent_work_queue<<<64, 256>>>(d_offsets, d_adj, d_out, n_items, d_global_idx);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel sync");

    cudaFree(d_offsets);
    cudaFree(d_adj);
    cudaFree(d_out);
    cudaFree(d_global_idx);
    return 0;
}