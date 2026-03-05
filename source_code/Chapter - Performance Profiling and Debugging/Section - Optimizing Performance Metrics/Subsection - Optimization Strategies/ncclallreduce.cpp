#include <nccl.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstdio>

#define CUDA_CHECK(call) do { cudaError_t e = call; if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e)); } while (0)
#define NCCL_CHECK(call) do { ncclResult_t r = call; if (r != ncclSuccess) throw std::runtime_error(std::string("NCCL: ") + ncclGetErrorString(r)); } while (0)

// RAII wrappers for automatic cleanup
struct CudaStreamDeleter { void operator()(cudaStream_t* s) const { if (s) { CUDA_CHECK(cudaStreamDestroy(*s)); delete s; } } };
struct CudaMemDeleter { void operator()(float* p) const { if (p) CUDA_CHECK(cudaFree(p)); } };
struct NcclCommDeleter { void operator()(ncclComm_t* c) const { if (c) { ncclCommDestroy(*c); delete c; } } };

using StreamPtr = std::unique_ptr<cudaStream_t, CudaStreamDeleter>;
using DevMemPtr = std::unique_ptr<float, CudaMemDeleter>;
using CommPtr   = std::unique_ptr<ncclComm_t, NcclCommDeleter>;

// Simple scale kernel
__global__ void scaleKernel(float* __restrict__ data, size_t n, float scale) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scale;
}

int main() {
    const int ndev = 4;
    const size_t grad_count = 1'000'000;

    std::vector<int> devs(ndev);
    for (int i = 0; i < ndev; ++i) devs[i] = i;

    std::vector<StreamPtr> streams;
    std::vector<DevMemPtr> d_grad;
    std::vector<CommPtr> comms;

    streams.reserve(ndev);
    d_grad.reserve(ndev);
    comms.reserve(ndev);

    // Allocate resources
    for (int i = 0; i < ndev; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));

        auto stream = std::make_unique<cudaStream_t>();
        CUDA_CHECK(cudaStreamCreateWithFlags(stream.get(), cudaStreamNonBlocking));
        streams.emplace_back(stream.release(), CudaStreamDeleter{});

        float* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, grad_count * sizeof(float)));
        d_grad.emplace_back(ptr, CudaMemDeleter{});

        auto comm = std::make_unique<ncclComm_t>();
        comms.emplace_back(comm.release(), NcclCommDeleter{});
    }

    NCCL_CHECK(ncclCommInitAll(reinterpret_cast<ncclComm_t*>(comms.data()), ndev, devs.data()));

    // AllReduce + scale
    for (int i = 0; i < ndev; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));
        NCCL_CHECK(ncclAllReduce(d_grad[i].get(), d_grad[i].get(), grad_count,
                                 ncclFloat, ncclSum, *comms[i], *streams[i]));

        const int block = 256;
        const int grid  = (grad_count + block - 1) / block;
        scaleKernel<<<grid, block, 0, *streams[i]>>>(d_grad[i].get(), grad_count, 1.0f / ndev);
    }

    // Sync
    for (int i = 0; i < ndev; ++i) {
        CUDA_CHECK(cudaSetDevice(devs[i]));
        CUDA_CHECK(cudaStreamSynchronize(*streams[i]));
    }

    return 0;
}