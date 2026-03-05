#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <thread>
#include <nvml.h>

int main(int argc, char** argv) {
    const unsigned int deviceIndex = 0;
    const unsigned int sampleMs = 50;
    const unsigned int durationMs = 10000;

    nvmlReturn_t result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::fprintf(stderr, "nvmlInit failed: %s\n", nvmlErrorString(result));
        return EXIT_FAILURE;
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex(deviceIndex, &device);
    if (result != NVML_SUCCESS) {
        std::fprintf(stderr, "nvmlDeviceGetHandleByIndex failed: %s\n", nvmlErrorString(result));
        nvmlShutdown();
        return EXIT_FAILURE;
    }

    std::vector<unsigned int> samples;
    samples.reserve(durationMs / sampleMs + 1);

    const auto start = std::chrono::steady_clock::now();
    const auto endTime = start + std::chrono::milliseconds(durationMs);

    while (std::chrono::steady_clock::now() < endTime) {
        unsigned int powerMw = 0;
        result = nvmlDeviceGetPowerUsage(device, &powerMw);
        if (result != NVML_SUCCESS) {
            std::fprintf(stderr, "nvmlDeviceGetPowerUsage failed: %s\n", nvmlErrorString(result));
            break;
        }
        samples.push_back(powerMw);
        std::this_thread::sleep_for(std::chrono::milliseconds(sampleMs));
    }

    double energyJ = 0.0;
    const double dtSec = sampleMs / 1000.0;
    for (unsigned int pMw : samples) energyJ += (pMw / 1000.0) * dtSec;

    const double avgPowerW = samples.empty() ? 0.0 : energyJ / (samples.size() * dtSec);
    std::printf("Samples: %zu, Duration: %.2f s\n", samples.size(), samples.size() * dtSec);
    std::printf("Energy: %.4f J, Avg Power: %.4f W\n", energyJ, avgPowerW);

    nvmlShutdown();
    return EXIT_SUCCESS;
}