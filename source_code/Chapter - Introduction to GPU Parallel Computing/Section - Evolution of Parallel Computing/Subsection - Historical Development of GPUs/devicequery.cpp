#include <cuda_runtime.h>
#include <iostream>
#include <cstdint>

// Return CUDA cores per SM for a given compute capability; 64 if unknown
int coresPerSM(int major, int minor) {
    struct Map { int major, minor, cores; };
    static const Map tbl[] = {
        {1,0,8},{1,1,8},{1,2,8},{1,3,8},
        {2,0,32},{2,1,48},
        {3,0,192},{3,5,192},
        {5,0,128},{5,2,128},
        {6,0,64},{6,1,128},{6,2,128},
        {7,0,64},{7,5,64},
        {8,0,64},{8,6,64},
        {9,0,128}  // Hopper
    };
    for (const auto& m : tbl)
        if (m.major == major && m.minor == minor) return m.cores;
    return 64;
}

int main() {
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    for (int d = 0; d < ndev; ++d) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p, d);

        int cores = coresPerSM(p.major, p.minor);
        double tflops = static_cast<double>(p.multiProcessorCount) *
                        cores * 2.0 *
                        static_cast<double>(p.clockRate) / 1e9;

        std::cout << "Device " << d << ": " << p.name << '\n'
                  << "  CC: " << p.major << '.' << p.minor
                  << "  SMs: " << p.multiProcessorCount
                  << "  cores/SM: " << cores << '\n'
                  << "  Clock: " << p.clockRate << " kHz"
                  << "  Memory: " << (p.totalGlobalMem >> 20) << " MiB\n"
                  << "  FP32 TFLOPS (theoretical): " << tflops << "\n\n";
    }
    return 0;
}