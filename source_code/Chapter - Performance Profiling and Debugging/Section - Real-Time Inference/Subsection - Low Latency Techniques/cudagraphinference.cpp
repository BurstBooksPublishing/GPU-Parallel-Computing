#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <memory>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            std::cerr << __FILE__ << ":" << __LINE__ << " CUDA "            \
                      << #call << " failed: " << cudaGetErrorString(err)    \
                      << "\n";                                              \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

__global__ void infer_kernel(const float* __restrict__ in,
                              float* __restrict__ out,
                              int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = in[idx] * 0.125f + 0.01f;
        out[idx] = tanhf(v);
    }
}

int main()
{
    constexpr int N = 1 << 20;
    constexpr size_t bytes = N * sizeof(float);
    constexpr int REPS = 1000;

    // pinned host buffers
    float *h_in = nullptr, *h_out = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_in, bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_out, bytes, cudaHostAllocPortable));

    // init input
    for (int i = 0; i < N; ++i) h_in[i] = (i & 255) * 0.001f;

    // device buffers
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // stream & graph
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in, bytes, cudaMemcpyHostToDevice, stream));
    constexpr int block = 256;
    const int grid = (N + block - 1) / block;
    infer_kernel<<<grid, block, 0, stream>>>(d_in, d_out, N);
    CUDA_CHECK(cudaMemcpyAsync(h_out, d_out, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // warmup
    CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // benchmark
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < REPS; ++i) {
        CUDA_CHECK(cudaGraphLaunch(graphExec, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "Avg latency per replay (ms): " << (ms / REPS) << "\n";

    // cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaStreamDestroy(stream);
    return 0;
}