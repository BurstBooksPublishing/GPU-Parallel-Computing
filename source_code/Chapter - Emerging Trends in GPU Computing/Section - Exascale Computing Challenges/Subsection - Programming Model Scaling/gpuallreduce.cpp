#include <cstdio>
#include <cstdlib>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CHECK_MPI(cmd) do { int e = cmd; if (e != MPI_SUCCESS) { fprintf(stderr,"MPI error %d\n",e); MPI_Abort(MPI_COMM_WORLD,e); }} while(0)
#define CHECK_CUDA(cmd) do { cudaError_t e = cmd; if (e != cudaSuccess) { fprintf(stderr,"CUDA error %s\n",cudaGetErrorString(e)); exit(1);} } while(0)
#define CHECK_NCCL(cmd) do { ncclResult_t r = cmd; if (r != ncclSuccess) { fprintf(stderr,"NCCL error %s\n",ncclGetErrorString(r)); MPI_Abort(MPI_COMM_WORLD,-1);} } while(0)

int main(int argc, char* argv[]) {
  CHECK_MPI(MPI_Init(&argc, &argv));
  int world_rank, world_size;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &world_size));

  int device;
  CHECK_CUDA(cudaGetDeviceCount(&device));
  device = world_rank % device;               // round-robin GPUs on node
  CHECK_CUDA(cudaSetDevice(device));
  cudaStream_t stream;
  CHECK_CUDA(cudaStreamCreate(&stream));

  ncclUniqueId id;
  ncclComm_t comm;
  if (world_rank == 0) CHECK_NCCL(ncclGetUniqueId(&id));
  CHECK_MPI(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  CHECK_NCCL(ncclCommInitRank(&comm, world_size, id, world_rank));

  const size_t count = 1ULL << 20;            // 1 M floats
  float* d_buf;
  CHECK_CUDA(cudaMalloc(&d_buf, count * sizeof(float)));

  std::vector<float> h_buf(count, 1.0f + world_rank);
  CHECK_CUDA(cudaMemcpyAsync(d_buf, h_buf.data(), count * sizeof(float),
                             cudaMemcpyHostToDevice, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));

  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  double t0 = MPI_Wtime();
  CHECK_NCCL(ncclAllReduce(d_buf, d_buf, count, ncclFloat, ncclSum, comm, stream));
  CHECK_CUDA(cudaStreamSynchronize(stream));
  double t1 = MPI_Wtime();

  if (world_rank == 0) {
    const double bytes = count * sizeof(float) * world_size;
    printf("Allreduce %zu floats on %d ranks: %.6f s  %.3f GB/s\n",
           count, world_size, t1 - t0, bytes / (t1 - t0) / 1e9);
  }

  CHECK_NCCL(ncclCommDestroy(comm));
  CHECK_CUDA(cudaFree(d_buf));
  CHECK_CUDA(cudaStreamDestroy(stream));
  CHECK_MPI(MPI_Finalize());
  return 0;
}