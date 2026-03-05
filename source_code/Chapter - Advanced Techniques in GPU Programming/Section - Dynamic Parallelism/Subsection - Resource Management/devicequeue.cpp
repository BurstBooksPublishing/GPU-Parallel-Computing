#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

#define CUDA_CHECK(call) do { cudaError_t e = call; if(e != cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

constexpr uint32_t QUEUE_CAPACITY = 1u << 16;
constexpr uint32_t MASK           = QUEUE_CAPACITY - 1u;

struct Task { int start; int length; };

__device__ uint32_t d_head;
__device__ uint32_t d_tail;
__device__ Task     d_buf[QUEUE_CAPACITY];

__device__ bool enqueue_task(Task t) {
    uint32_t p = atomicAdd(&d_tail, 1u);
    if (p - atomicAdd(&d_head, 0u) >= QUEUE_CAPACITY) {   // full
        atomicSub(&d_tail, 1u);
        return false;
    }
    d_buf[p & MASK] = t;
    return true;
}

__global__ void persistent_workers(volatile bool* done) {
    while (!*done) {
        uint32_t h = atomicAdd(&d_head, 0u);
        uint32_t t = atomicAdd(&d_tail, 0u);
        if (h == t) { __nanosleep(100); continue; }

        Task task = d_buf[atomicAdd(&d_head, 1u) & MASK];

        float acc = 0.0f;
        for (int i = 0; i < task.length; ++i) acc += __fadd_rn(task.start + i, 0.0f);
    }
}

__global__ void producer_enqueue(int nTasks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nTasks) return;
    Task t{idx * 1024, 1024};
    for (int r = 0; r < 16 && !enqueue_task(t); ++r) __nanosleep(50);
}

int main() {
    CUDA_CHECK(cudaSetDevice(0));

    bool* d_done;
    CUDA_CHECK(cudaMalloc(&d_done, sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_done, 0, sizeof(bool)));

    CUDA_CHECK(cudaMemset(&d_head, 0, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(&d_tail, 0, sizeof(uint32_t)));

    dim3 wkB(128), wkG(64);
    persistent_workers<<<wkG, wkB>>>(d_done);
    CUDA_CHECK(cudaGetLastError());

    int nTasks = 1 << 14;
    dim3 pb(256), pg((nTasks + pb.x - 1) / pb.x);
    producer_enqueue<<<pg, pb>>>(nTasks);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool h_done = true;
    CUDA_CHECK(cudaMemcpy(d_done, &h_done, sizeof(bool), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_done));
    return 0;
}