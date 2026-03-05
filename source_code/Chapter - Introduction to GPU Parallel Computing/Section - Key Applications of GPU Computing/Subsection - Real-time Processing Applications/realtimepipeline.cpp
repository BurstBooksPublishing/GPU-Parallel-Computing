#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstring>

#define CUDA_CHECK(call) do { \
  cudaError_t e = (call); if (e != cudaSuccess) { \
    throw std::runtime_error(cudaGetErrorString(e)); }} while(0)

__global__ void invert_colors_kernel(unsigned char* d_img, int pixels) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < pixels) {
    d_img[3*idx + 0] = 255 - d_img[3*idx + 0];
    d_img[3*idx + 1] = 255 - d_img[3*idx + 1];
    d_img[3*idx + 2] = 255 - d_img[3*idx + 2];
  }
}

int main() {
  const int width = 1920, height = 1080;
  const int pixels = width * height;
  const size_t frame_bytes = size_t(pixels) * 3; // RGB8
  const int frames = 1000;

  unsigned char *h_buf[2];
  CUDA_CHECK(cudaMallocHost(&h_buf[0], frame_bytes));
  CUDA_CHECK(cudaMallocHost(&h_buf[1], frame_bytes));

  unsigned char *d_buf[2];
  CUDA_CHECK(cudaMalloc(&d_buf[0], frame_bytes));
  CUDA_CHECK(cudaMalloc(&d_buf[1], frame_bytes));

  cudaStream_t streams[2];
  CUDA_CHECK(cudaStreamCreate(&streams[0]));
  CUDA_CHECK(cudaStreamCreate(&streams[1]));

  dim3 block(256);
  dim3 grid((pixels + block.x - 1) / block.x);

  auto start = std::chrono::high_resolution_clock::now();
  for (int f = 0; f < frames; ++f) {
    int i = f & 1;
    int j = 1 - i;

    memset(h_buf[i], (unsigned char)(f & 255), frame_bytes);

    CUDA_CHECK(cudaMemcpyAsync(d_buf[i], h_buf[i], frame_bytes,
                               cudaMemcpyHostToDevice, streams[i]));

    invert_colors_kernel<<<grid, block, 0, streams[i]>>>(d_buf[i], pixels);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(h_buf[i], d_buf[i], frame_bytes,
                               cudaMemcpyDeviceToHost, streams[i]));

    if (f > 0) CUDA_CHECK(cudaStreamSynchronize(streams[j]));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end - start;
  std::cout << "Processed " << frames << " frames in "
            << elapsed.count() << " s (" << frames/elapsed.count() << " FPS)\n";

  for (int k = 0; k < 2; ++k) {
    cudaFree(d_buf[k]);
    cudaFreeHost(h_buf[k]);
    cudaStreamDestroy(streams[k]);
  }
  return 0;
}