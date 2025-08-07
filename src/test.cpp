// @file test.cpp
// @brief Test program for the MPPI solver.
// @author MC00614

#include "mppi_solver.cuh"

#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // Problem parameters
    const int N = 8000;
    const int T = 5;
    const double dt = 0.02;
    const double gamma = 5.0;

    // Problem specific parameters
    double anchor[24] = {
        0,    0,    0,
        0,    8,    0,
        8.86, 8.0,  0,
        8.86, 0,    0,
        0,    0,    2.2,
        0,    8,    2.2,
        8.86, 8.0,  2.2,
        8.86, 0,    2.2
    };
    double gravity[3] = {0, 0, -9.81};
    double sigma[6] = {5.0, 5.0, 5.0, 1.0, 1.0, 1.0};
    MPPISolver solver(N, T, dt, gamma, anchor, gravity, sigma);

    // Initial state and control inputs
    double x0[12] = {};
    std::vector<double> U0(6 * T, 0.0);
    std::vector<double> Uopt(6 * T, 0.0);
    // Problem specific
    std::vector<double> ranges(T * 8, 5.0);
    std::vector<double> accgyr(T * 6, 0.0);

    auto start = std::chrono::high_resolution_clock::now();
    solver.solve(x0, NULL, U0.data(), ranges.data(), accgyr.data(), Uopt.data());
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "solve() time: " << elapsed.count() * 1000.0 << " ms\n";

    std::cout << "u0 =";
    for (int k = 0; k < 6; ++k) {
        std::cout << " " << Uopt[k];
    }
    std::cout << std::endl;

    return 0;
}
