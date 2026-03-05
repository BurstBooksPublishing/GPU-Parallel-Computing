#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(err) do { cudaError_t e = (err); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

template<typename T, int TILE>
__global__ void tiled_gemm(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                           int M, int N, int K, T alpha, T beta) {
    __shared__ T As[TILE][TILE];
    __shared__ T Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    T sum = (T)0;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        int b_row = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : (T)0;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : (T)0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

int main() {
    const int M = 512, N = 512, K = 512;
    using scalar_t = float;
    const size_t SIZE_A = M*K, SIZE_B = K*N, SIZE_C = M*N;
    scalar_t *hA = (scalar_t*)malloc(SIZE_A*sizeof(scalar_t));
    scalar_t *hB = (scalar_t*)malloc(SIZE_B*sizeof(scalar_t));
    scalar_t *hC = (scalar_t*)malloc(SIZE_C*sizeof(scalar_t));
    for (size_t i=0;i<SIZE_A;i++) hA[i] = (scalar_t)(rand()/(scalar_t)RAND_MAX);
    for (size_t i=0;i<SIZE_B;i++) hB[i] = (scalar_t)(rand()/(scalar_t)RAND_MAX);
    scalar_t *dA,*dB,*dC;
    CUDA_CHECK(cudaMalloc(&dA,SIZE_A*sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&dB,SIZE_B*sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&dC,SIZE_C*sizeof(scalar_t)));
    CUDA_CHECK(cudaMemcpy(dA,hA,SIZE_A*sizeof(scalar_t),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB,hB,SIZE_B*sizeof(scalar_t),cudaMemcpyHostToDevice));
    dim3 block(16,16);
    dim3 grid((N+block.x-1)/block.x,(M+block.y-1)/block.y);
    tiled_gemm<scalar_t,16><<<grid,block>>>(dA,dB,dC,M,N,K,1.0f,0.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC,dC,SIZE_C*sizeof(scalar_t),cudaMemcpyDeviceToHost));

    printf("C[0]=%f\n", hC[0]); // simple correctness check

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}