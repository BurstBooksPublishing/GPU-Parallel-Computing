#include <iostream>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>

static const std::unordered_map<std::string, int> kCoresPerSM = {
    {"A100", 64}, {"V100", 64}, {"TITAN", 128}, {"RTX 20", 64}, {"RTX 30", 128}
};

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    for (int dev = 0; dev < dev_count; ++dev) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) continue;

        std::string name = prop.name;
        int sm_count = prop.multiProcessorCount;
        double clock_hz = prop.clockRate * 1000.0;
        double mem_clock_hz = prop.memoryClockRate * 1000.0;
        int bus_width = prop.memoryBusWidth;

        int cores_sm = 64; // default
        for (const auto& kv : kCoresPerSM)
            if (name.find(kv.first) != std::string::npos) { cores_sm = kv.second; break; }

        double peak_flops = sm_count * cores_sm * clock_hz * 2.0; // 2 FMA ops/cycle
        double bandwidth_gb_s = (mem_clock_hz * (bus_width / 8.0)) / 1e9;

        std::cout << "Device " << dev << ": " << name << '\n'
                  << "  SMs=" << sm_count << ", cores/SM=" << cores_sm
                  << ", core clock=" << (clock_hz / 1e6) << " MHz\n"
                  << "  Peak FP32 FLOPS = " << (peak_flops / 1e12) << " TFLOPS\n"
                  << "  Memory bandwidth ≈ " << bandwidth_gb_s << " GB/s\n\n";
    }
    return 0;
}