#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <cuda_runtime.h>

#define CHECK_MPI(call) \
    do { \
        int e = (call); \
        if (e != MPI_SUCCESS) { \
            fprintf(stderr, "MPI error %d at %s:%d\n", e, __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

#define CHECK_CUDA(call) \
    do { \
        cudaError_t e = (call); \
        if (e != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main(int argc, char **argv) {
    CHECK_MPI(MPI_Init(&argc, &argv));
    int rank, nprocs;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &nprocs));
    if (nprocs != 2) {
        if (rank == 0) fprintf(stderr, "Requires exactly 2 ranks\n");
        CHECK_MPI(MPI_Abort(MPI_COMM_WORLD, 1));
    }

    const size_t N = 1 << 20;               // 1 MiB
    CHECK_CUDA(cudaSetDevice(rank));        // multi-GPU safety
    void *d_buf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_buf, N));

    MPI_Win win;
    CHECK_MPI(MPI_Win_create(d_buf, N, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win));

    if (rank == 0) {
        void *d_src = nullptr;
        CHECK_CUDA(cudaMalloc(&d_src, N));
        CHECK_CUDA(cudaMemset(d_src, 0xAA, N));

        CHECK_MPI(MPI_Win_fence(0, win));
        CHECK_MPI(MPI_Put(d_src, N, MPI_BYTE, 1, 0, N, MPI_BYTE, win));
        CHECK_MPI(MPI_Win_fence(0, win));

        CHECK_CUDA(cudaFree(d_src));
    } else {
        CHECK_MPI(MPI_Win_fence(0, win));
        CHECK_MPI(MPI_Win_fence(0, win));
    }

    CHECK_MPI(MPI_Win_free(&win));
    CHECK_CUDA(cudaFree(d_buf));
    CHECK_MPI(MPI_Finalize());
    return 0;
}