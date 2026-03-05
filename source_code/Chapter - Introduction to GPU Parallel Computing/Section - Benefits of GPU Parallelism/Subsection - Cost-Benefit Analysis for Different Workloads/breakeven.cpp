#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::cerr << "Usage: " << argv[0] << " T_cpu H_cpu E_cpu H_gpu E_gpu D_cpu D_gpu\n";
        return EXIT_FAILURE;
    }

    try {
        const double T_cpu  = std::stod(argv[1]);
        const double H_cpu  = std::stod(argv[2]);
        const double E_cpu  = std::stod(argv[3]);
        const double H_gpu  = std::stod(argv[4]);
        const double E_gpu  = std::stod(argv[5]);
        const double D_cpu  = std::stod(argv[6]);
        const double D_gpu  = std::stod(argv[7]);

        const double C_cpu = (H_cpu + E_cpu) * T_cpu + D_cpu;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "CPU cost per job: $" << C_cpu << '\n';

        const double denom = C_cpu - D_gpu;
        if (denom <= 0.0) {
            std::cout << "GPU requires amortization or lower dev cost; denom <= 0\n";
            return EXIT_SUCCESS;
        }

        const double S_req = (H_gpu + E_gpu) * T_cpu / denom;
        std::cout << "Required speedup S to break even: " << S_req << "x\n";
        std::cout << "GPU cost at break-even: $" << ((H_gpu + E_gpu) * T_cpu / S_req + D_gpu) << '\n';
    } catch (const std::exception&) {
        std::cerr << "Invalid numeric input\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}