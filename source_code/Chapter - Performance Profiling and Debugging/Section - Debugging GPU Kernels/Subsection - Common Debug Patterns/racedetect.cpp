#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

static inline void checkCuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void provoke_race(int *slots, int n_slots, int iters_per_thread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = 0; i < iters_per_thread; ++i) {
        int s = (tid + i) % n_slots;
        // Non-atomic increment intentionally to provoke race in debug runs.
        slots[s] = slots[s] + 1;
    }
}

int main() {
    const int threads = 1024 * 4;
    const int blocks = 64;
    const int n_slots = 16;                // small to force conflicting writes
    const int iters_per_thread = 1000;
    const int runs = 2000;                 // sampling iterations

    int *d_slots;
    size_t slots_bytes = n_slots * sizeof(int);
    checkCuda(cudaMalloc(&d_slots, slots_bytes), "cudaMalloc slots");

    for (int run = 0; run < runs; ++run) {
        checkCuda(cudaMemset(d_slots, 0, slots_bytes), "cudaMemset slots");
        provoke_race<<<blocks, threads>>>(d_slots, n_slots, iters_per_thread);
        checkCuda(cudaGetLastError(), "kernel launch");
        checkCuda(cudaDeviceSynchronize(), "synchronize");

        int h_slots[n_slots];
        checkCuda(cudaMemcpy(h_slots, d_slots, slots_bytes, cudaMemcpyDeviceToHost),
                  "memcpy slots");

        int expected = (threads * blocks * iters_per_thread) / n_slots;
        bool bad = false;
        for (int i = 0; i < n_slots; ++i) {
            if (h_slots[i] != expected) { bad = true; break; }
        }
        if (bad) {
            std::printf("Race detected on run %d; sample slot values:", run);
            for (int i = 0; i < n_slots; ++i) std::printf(" %d", h_slots[i]);
            std::printf("\n");
            break;
        }
    }

    cudaFree(d_slots);
    return 0;
}