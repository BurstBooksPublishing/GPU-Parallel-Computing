#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>

#define CHECK_MPI(call)                                                  \
    do {                                                                 \
        int e = (call);                                                  \
        if (e != MPI_SUCCESS) {                                          \
            char errstr[MPI_MAX_ERROR_STRING];                           \
            int len;                                                     \
            MPI_Error_string(e, errstr, &len);                           \
            fprintf(stderr, "MPI error: %s\n", errstr);                  \
            MPI_Abort(MPI_COMM_WORLD, e);                                \
        }                                                                \
    } while (0)

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t e = (call);                                          \
        if (e != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

int main(int argc, char** argv) {
    CHECK_MPI(MPI_Init(&argc, &argv));

    int rank, size;
    CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "This example requires exactly 2 MPI ranks.\n");
        CHECK_MPI(MPI_Finalize());
        return EXIT_FAILURE;
    }

    const size_t N = 1 << 20;               // 1M floats
    const size_t bytes = N * sizeof(float);

    CHECK_CUDA(cudaSetDevice(rank));
    float* d_buf;
    CHECK_CUDA(cudaMalloc(&d_buf, bytes));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Fill buffer with rank-dependent pattern
    CHECK_CUDA(cudaMemsetAsync(d_buf, rank, bytes, stream));

    int peer = 1 - rank;
    MPI_Request reqs[2];
    CHECK_MPI(MPI_Irecv(d_buf, N, MPI_FLOAT, peer, 0, MPI_COMM_WORLD, &reqs[0]));
    CHECK_MPI(MPI_Isend(d_buf, N, MPI_FLOAT, peer, 0, MPI_COMM_WORLD, &reqs[1]));

    CHECK_MPI(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Verify received data
    float host_sample;
    CHECK_CUDA(cudaMemcpyAsync(&host_sample, d_buf + N / 2, sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    printf("Rank %d received sample value %f\n", rank, host_sample);

    CHECK_CUDA(cudaFree(d_buf));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_MPI(MPI_Finalize());
    return EXIT_SUCCESS;
}