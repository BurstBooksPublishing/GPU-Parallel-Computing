c
#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define CHECK_CU(call) do { CUresult r = (call); if (r != CUDA_SUCCESS) { \
    const char *msg; cuGetErrorName(r, &msg); \
    fprintf(stderr, "CU error %s:%d: %s\n", __FILE__, __LINE__, msg); exit(EXIT_FAILURE); }} while(0)

int main(void) {
    CHECK_CU(cuInit(0));
    CUdevice dev; CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, dev));

    const char *ptx =
        ".version 6.0\n"
        ".target sm_50\n"
        ".address_size 64\n"
        ".visible .entry empty_kernel() { ret; }\n";

    CUmodule mod; CHECK_CU(cuModuleLoadData(&mod, ptx));
    CUfunction func; CHECK_CU(cuModuleGetFunction(&func, mod, "empty_kernel"));

    const int N = 100000; // more iterations for stable timing
    struct timespec t0, t1;
    CHECK_CU(cuCtxSynchronize());
    clock_gettime(CLOCK_MONOTONIC_RAW, &t0);

    for (int i = 0; i < N; ++i) {
        CHECK_CU(cuLaunchKernel(func, 1,1,1, 1,1,1, 0, 0, NULL, NULL));
    }
    CHECK_CU(cuCtxSynchronize());
    clock_gettime(CLOCK_MONOTONIC_RAW, &t1);

    double elapsed_us = (t1.tv_sec - t0.tv_sec)*1e6 + (t1.tv_nsec - t0.tv_nsec)*1e-3;
    printf("Average launch latency: %.3f microseconds\n", elapsed_us / N);

    CHECK_CU(cuModuleUnload(mod));
    CHECK_CU(cuCtxDestroy(ctx));
    return EXIT_SUCCESS;
}