#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#define CUDA_CHECK(call) do {                                    \
  cudaError_t e = (call); if (e != cudaSuccess) {                \
    fprintf(stderr,"CUDA Error %s:%d: %s\n",__FILE__,__LINE__,   \
            cudaGetErrorString(e)); exit(1); } } while(0)

constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;

__global__ void stencil3d(const float* __restrict__ a, float* __restrict__ b,
                          int Nx, int Ny, int Nz, float alpha) {
  __shared__ float s[TILE_Y+2][TILE_X+2];
  int gx = blockIdx.x * TILE_X + threadIdx.x;
  int gy = blockIdx.y * TILE_Y + threadIdx.y;
  int z  = blockIdx.z;
  int lx = threadIdx.x + 1;
  int ly = threadIdx.y + 1;

  if (gx < Nx && gy < Ny && z < Nz)
    s[ly][lx] = a[(z*Ny + gy)*Nx + gx];
  else
    s[ly][lx] = 0.0f;

  if (threadIdx.x == 0) {
    int gx_m = gx - 1;
    s[ly][0] = (gx_m>=0 && gy<Ny && z<Nz) ? a[(z*Ny+gy)*Nx+gx_m] : 0.0f;
  }
  if (threadIdx.x == TILE_X-1) {
    int gx_p = gx + 1;
    s[ly][lx+1] = (gx_p<Nx && gy<Ny && z<Nz) ? a[(z*Ny+gy)*Nx+gx_p] : 0.0f;
  }
  if (threadIdx.y == 0) {
    int gy_m = gy - 1;
    s[0][lx] = (gx<Nx && gy_m>=0 && z<Nz) ? a[(z*Ny+gy_m)*Nx+gx] : 0.0f;
  }
  if (threadIdx.y == TILE_Y-1) {
    int gy_p = gy + 1;
    s[ly+1][lx] = (gx<Nx && gy_p<Ny && z<Nz) ? a[(z*Ny+gy_p)*Nx+gx] : 0.0f;
  }
  __syncthreads();

  if (gx < Nx && gy < Ny && z < Nz) {
    float c = s[ly][lx];
    float v = (s[ly][lx-1] + s[ly][lx+1] +
               s[ly-1][lx] + s[ly+1][lx] +
               c) * alpha;
    if (z > 0)   v += a[((z-1)*Ny + gy)*Nx + gx] * alpha;
    if (z+1 < Nz) v += a[((z+1)*Ny + gy)*Nx + gx] * alpha;
    b[(z*Ny + gy)*Nx + gx] = v;
  }
}

int main(int argc, char** argv) {
  int Nx = 256, Ny = 256, Nz = 256;
  if (argc >= 4) {
    Nx = std::atoi(argv[1]);
    Ny = std::atoi(argv[2]);
    Nz = std::atoi(argv[3]);
  }
  float alpha = 0.16666667f;
  size_t N = size_t(Nx)*Ny*Nz;
  std::vector<float> h_a(N);
  std::generate(h_a.begin(), h_a.end(), [n=0]() mutable { return sinf((n++)*0.01f); });

  float *d_a, *d_b;
  CUDA_CHECK(cudaMalloc(&d_a, N*sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_b, N*sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_a,h_a.data(),N*sizeof(float),cudaMemcpyHostToDevice));

  dim3 block(TILE_X,TILE_Y,1);
  dim3 grid((Nx+TILE_X-1)/TILE_X,(Ny+TILE_Y-1)/TILE_Y,Nz);

  cudaEvent_t t0,t1;
  CUDA_CHECK(cudaEventCreate(&t0));
  CUDA_CHECK(cudaEventCreate(&t1));
  CUDA_CHECK(cudaEventRecord(t0));
  stencil3d<<<grid,block>>>(d_a,d_b,Nx,Ny,Nz,alpha);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(t1));
  CUDA_CHECK(cudaEventSynchronize(t1));
  float ms=0;
  CUDA_CHECK(cudaEventElapsedTime(&ms,t0,t1));
  double bytes = double(N)*sizeof(float)*2.0;
  double bw = (bytes / (ms/1000.0)) / 1e9;
  printf("Size %dx%dx%d, time %.3f ms, effective BW %.2f GB/s\n",Nx,Ny,Nz,ms,bw);
  CUDA_CHECK(cudaMemcpy(h_a.data(),d_b,N*sizeof(float),cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  return 0;
}