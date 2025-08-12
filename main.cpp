// @file main.cpp
// @brief Main program for the MPPI solver with IMU and UWB data processing.
// @author MC00614

#include "helper.h"

#include "mppi_solver.cuh"

#include <iostream>
#include <vector>
#include <chrono>

#include <iostream>

#include <algorithm>

int main() {
    const std::string imuPath     = "../data/imu_test2.txt";
    const std::string uwbPath    = "../data/uwb_test2.txt";
    const std::string anchorsPath= "../config/anchors_0620.txt";

    std::vector<ImuData> imuAll;
    std::vector<UwbData> uwbLeft, uwbRight;
    if(!loadImu(imuPath, imuAll) || !loadUwb(uwbPath, uwbLeft, uwbRight))
        return -1;

    double imuOffset = imuAll.front().timeStamp;
    for(auto &d : imuAll) d.timeStamp -= imuOffset;
    double uwbOffset = std::min(uwbLeft.front().timeStamp, uwbRight.front().timeStamp);
    for(auto &d : uwbLeft)  d.timeStamp -= uwbOffset;
    for(auto &d : uwbRight) d.timeStamp -= uwbOffset;

    // ========== MPPI Solver ==========
    // Problem parameters
    const int N = 2000;
    const int T = 5;
    const double dt = 0.02;
    const double gamma = 5.0;

    // Problem specific parameters
    double anchor[24];
    std::ifstream ifs(anchorsPath);
    for (int i = 0; i < 24; ++i) ifs >> anchor[i];
    double gravity[3] = {0, 0, -9.81};
    double sigma[6] = {10.0, 10.0, 10.0, 1.0, 1.0, 1.0};
    MPPISolver solver(N, T, dt, gamma, anchor, gravity, sigma);
    
    std::ofstream poseFile("../data/mppi_pose_2.txt");
    if(!poseFile) { std::cerr << "Failed to open pose.txt" << std::endl; return -1; }
    poseFile << "# time x y z qx qy qz qw\n";
    
    size_t imuIdx = 0, ul = 0, ur = 0;
    double prevTs = 0;

    // Initial state and control inputs
    double x0[15] = {0.0};
    x0[6] = 1.0;
    x0[10] = 1.0;
    x0[14] = 1.0;
    double U0[6 * T] = {0.0};

    // Storage for next state
    double xn[15] = {0.0};
    // Storage for optimized control inputs
    double Uopt[6 * T] = {0.0};

    auto total_start = std::chrono::high_resolution_clock::now();

    int iter = 0;
    double solve_duration = 0.0;
    while (true) {
        std::vector<ImuData> imuBatch;
        std::vector<UwbData> uwbBatch;
        if(!getNextBatch(T, imuAll, imuIdx,
                         uwbLeft, ul, uwbRight, ur,
                         imuBatch, uwbBatch)) break;
        double ts = uwbBatch.front().timeStamp;
        if(prevTs > 0) solver.setDt(ts - prevTs);
        prevTs = ts;

        double ranges[16 * T];
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < 16; ++i) {
                ranges[t * 16 + i] = uwbBatch[t].ranges(i);
            }
        }
        double accgyr[6 * T];
        for (int t = 0; t < T; ++t) {
            accgyr[t * 6 + 0] = imuBatch[t].acc.x();
            accgyr[t * 6 + 1] = imuBatch[t].acc.y();
            accgyr[t * 6 + 2] = imuBatch[t].acc.z();
            accgyr[t * 6 + 3] = imuBatch[t].gyr.x();
            accgyr[t * 6 + 4] = imuBatch[t].gyr.y();
            accgyr[t * 6 + 5] = imuBatch[t].gyr.z();
        }

        auto solve_start = std::chrono::high_resolution_clock::now();

        solver.solve(Uopt, xn, x0, U0, ranges, accgyr);
        cudaDeviceSynchronize();

        auto solve_end = std::chrono::high_resolution_clock::now();
        solve_duration += std::chrono::duration<double, std::milli>(solve_end - solve_start).count();

        // Update initial state for next iteration
        std::copy(xn, xn + 15, x0);
        // Warm start control inputs
        std::copy(Uopt + 6, Uopt + 6 * T, U0);
        std::copy(Uopt + 6 * (T - 1), Uopt + 6 * T, U0 + 6 * (T - 1));

        // Iteration count
        iter++;

        // Log current state
        STATE st(xn);
        poseFile << ts << " "
             << st.p.x() << " "
             << st.p.y() << " "
             << st.p.z() << " ";
        Eigen::Quaterniond q(st.R);
        poseFile << q.x() << " "
                 << q.y() << " "
                 << q.z() << " "
                 << q.w() << "\n";
                 std::cout<< ts <<" pos=["<< st.p.transpose() <<"]\n";

    }
    auto total_end = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    std::cout << "\nTotal time: " << totalTime << " ms" << std::endl;
    std::cout << "Average time per iteration: " << totalTime / iter << " ms" << std::endl;
    std::cout << "\nTotal solve time: " << solve_duration << " ms" << std::endl;
    std::cout << "Average solve time per iteration: " << solve_duration / iter << " ms" << std::endl;
    std::cout << "" << std::endl;

    poseFile.close();

    int ret = std::system("python3 ../scripts/plot.py"); (void)ret;


    return 0;
}
