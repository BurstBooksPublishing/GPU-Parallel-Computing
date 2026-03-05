#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>

#define CHECK_MPI(call)                                                                 \
    do {                                                                                \
        int e = (call);                                                                 \
        if (e != MPI_SUCCESS) {                                                         \
            char s[MPI_MAX_ERROR_STRING];                                               \
            int l;                                                                      \
            MPI_Error_string(e, s, &l);                                                 \
            throw std::runtime_error(std::string("MPI: ") + s);                         \
        }                                                                               \
    } while (0)

#define CHECK_CUDA(call)                                                                \
    do {                                                                                \
        cudaError_t e = (call);                                                         \
        if (e != cudaSuccess)                                                           \
            throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(e));    \
    } while (0)

#define CHECK_NCCL(call)                                                                \
    do {                                                                                \
        ncclResult_t r = (call);                                                        \
        if (r != ncclSuccess)                                                           \
            throw std::runtime_error(std::string("NCCL: ") + ncclGetErrorString(r));    \
    } while (0)

int main(int argc, char** argv) {
    CHECK_MPI(MPI_Init(&argc, &argv));

    int world_size, world_rank;
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

    int local_gpu_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&local_gpu_count));
    if (local_gpu_count == 0) throw std::runtime_error("No CUDA devices found");

    int local_rank = world_rank % local_gpu_count; // homogeneous node layout
    CHECK_CUDA(cudaSetDevice(local_rank));

    const std::size_t N = 1ULL << 24; // 16M floats
    float* d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, N * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_buf, 0, N * sizeof(float))); // zero-init

    cudaStream_t comp_stream{}, comm_stream{};
    CHECK_CUDA(cudaStreamCreateWithFlags(&comp_stream, cudaStreamNonBlocking));
    CHECK_CUDA(cudaStreamCreateWithFlags(&comm_stream, cudaStreamNonBlocking));

    ncclUniqueId id;
    if (world_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
    CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    ncclComm_t nccl_comm{};
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, world_size, id, world_rank));

    // dummy kernel to demonstrate compute-comm overlap
    constexpr int block = 256;
    const int grid = (N + block - 1) / block;
    increment_kernel<<<grid, block, 0, comp_stream>>>(d_buf, N, 1.0f);
    CHECK_CUDA(cudaGetLastError());

    CHECK_NCCL(ncclAllReduce(d_buf, d_buf, N, ncclFloat, ncclSum, nccl_comm, comm_stream));

    CHECK_CUDA(cudaStreamSynchronize(comm_stream));

    CHECK_NCCL(ncclCommDestroy(nccl_comm));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaStreamDestroy(comp_stream));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));
    CHECK_MPI(MPI_Finalize());

    if (world_rank == 0) std::cout << "AllReduce complete\n";
    return 0;
}

__global__ void increment_kernel(float* buf, std::size_t n, float val) {
    auto i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) buf[i] += val;
}