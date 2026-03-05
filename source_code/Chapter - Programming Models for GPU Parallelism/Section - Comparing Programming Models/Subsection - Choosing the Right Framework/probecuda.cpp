#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

int main() {
    int devCount = 0;
    checkCuda(cudaGetDeviceCount(&devCount), "cudaGetDeviceCount failed");

    if (devCount == 0) {
        std::cout << "No CUDA devices found; consider OpenCL/SYCL/Metal depending on platform.\n";
        return 0;
    }

    std::cout << "Detected " << devCount << " CUDA device(s).\n";

    for (int i = 0; i < devCount; ++i) {
        cudaDeviceProp prop{};
        checkCuda(cudaGetDeviceProperties(&prop, i), "cudaGetDeviceProperties failed");

        std::cout << "Device " << i << ": " << prop.name << '\n'
                  << "  Compute capability: " << prop.major << '.' << prop.minor << '\n'
                  << "  SMs: " << prop.multiProcessorCount
                  << ", Global mem: " << (prop.totalGlobalMem >> 20) << " MB\n";

        bool hasTensor = prop.major >= 7; // Volta+
        std::cout << "  Tensor cores: " << (hasTensor ? "yes" : "no") << '\n'
                  << "  Recommendation: "
                  << (hasTensor ? "CUDA + cuBLAS/cuDNN/TensorRT for ML and dense compute.\n"
                                : "CUDA or OpenCL/SYCL if portability required.\n");
    }
    return 0;
}