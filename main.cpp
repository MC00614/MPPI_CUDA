// @file main.cpp
// @brief Main program for the MPPI solver with IMU and UWB data processing.
// @author MC00614
// @cite KYH7238, mppi_estimation

#include "helper.h"
#include "mppi_solver.cuh"

#include <iostream>
#include <vector>
#include <chrono>

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstdlib>

bool loadImu(const std::string& path, std::vector<ImuData>& imuAll) {
    std::ifstream ifs(path);
    if(!ifs) { std::cerr << "Cannot open " << path << std::endl; return false; }
    std::string line;
    while(std::getline(ifs, line)) {
        if(line.empty()) continue;
        std::istringstream iss(line);
        ImuData d;
        iss >> d.timeStamp
            >> d.acc.x() >> d.acc.y() >> d.acc.z()
            >> d.gyr.x() >> d.gyr.y() >> d.gyr.z();
        imuAll.push_back(d);
    }
    return true;
}

bool loadUwb(const std::string& path,
             std::vector<UwbData>& uwbLeft,
             std::vector<UwbData>& uwbRight)
{
    std::ifstream ifs(path);
    if(!ifs) { std::cerr << "Cannot open " << path << std::endl; return false; }
    std::string line;
    while(std::getline(ifs, line)) {
        if(line.empty()) continue;
        std::istringstream iss(line);
        UwbData d;
        iss >> d.id >> d.timeStamp;
        d.ranges.resize(8);
        for(int i = 0; i < 8; ++i) iss >> d.ranges(i);
        if(d.id == 0) uwbLeft.push_back(d);
        else          uwbRight.push_back(d);
    }
    return true;
}

bool getNextBatch(int T,
    const std::vector<ImuData>& imuAll, size_t& imuIdx,
    const std::vector<UwbData>& uwbLeft, size_t& ulIdx,
    const std::vector<UwbData>& uwbRight, size_t& urIdx,
    std::vector<ImuData>& imuBatch,
    std::vector<UwbData>& uwbBatch)
{
    imuBatch.clear();
    uwbBatch.clear();
    while((int)uwbBatch.size() < T) {
        if(ulIdx >= uwbLeft.size() || urIdx >= uwbRight.size()) return false;
        double tsL = uwbLeft[ulIdx].timeStamp;
        double tsR = uwbRight[urIdx].timeStamp;
        if(std::abs(tsL - tsR) > 1e-6) {
            if(tsL < tsR) ++ulIdx;
            else           ++urIdx;
            continue;
        }
        double ts = tsL;
        while(imuIdx + 1 < imuAll.size() && imuAll[imuIdx+1].timeStamp < ts) ++imuIdx;
        if(imuIdx + 1 >= imuAll.size()) return false;
        const ImuData &a = imuAll[imuIdx];
        const ImuData &b = imuAll[imuIdx+1];
        double w = (ts - a.timeStamp) / (b.timeStamp - a.timeStamp);
        ImuData di;
        di.timeStamp = ts;
        di.acc = a.acc * (1-w) + b.acc * w;
        di.gyr = a.gyr * (1-w) + b.gyr * w;
        imuBatch.push_back(di);
        UwbData du;
        du.timeStamp = ts;
        du.id = 0;
        du.ranges.resize(16);
        du.ranges.head(8) = uwbLeft[ulIdx].ranges;
        du.ranges.tail(8) = uwbRight[urIdx].ranges;
        uwbBatch.push_back(du);
        ++ulIdx;
        ++urIdx;
    }
    return true;
}

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

    Eigen::Matrix<double,3,8> anchors;
    std::ifstream ifs(anchorsPath);
    for(int i = 0; i < 8; ++i)
        ifs >> anchors(0,i) >> anchors(1,i) >> anchors(2,i);

    // mppiEstimation mppi;
    // mppi.setAnchorPositions(anchors);

    // ========== MPPI Solver ==========
    // Problem parameters
    const int N = 8000;
    const int T = 3;
    const double dt = 0.02;
    const double gamma = 10.0;

    // Problem specific parameters
    double anchor[24];
    for (int i = 0; i < 8; ++i) {
        anchor[i * 3 + 0] = anchors(0, i);
        anchor[i * 3 + 1] = anchors(1, i);
        anchor[i * 3 + 2] = anchors(2, i);
    }
    double gravity[3] = {0, 0, -9.81};
    double sigma[6] = {15.0, 15.0, 15.0, 5.0, 5.0, 5.0};
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

    auto start = std::chrono::high_resolution_clock::now();

    while(true) {
        std::vector<ImuData> imuBatch;
        std::vector<UwbData> uwbBatch;
        if(!getNextBatch(T, imuAll, imuIdx,
                         uwbLeft, ul, uwbRight, ur,
                         imuBatch, uwbBatch)) break;
        double ts = uwbBatch.front().timeStamp;
        if(prevTs > 0) solver.setDt(ts - prevTs);
        prevTs = ts;

        double ranges[8 * T];
        for (int t = 0; t < T; ++t) {
            for (int i = 0; i < 8; ++i) {
                ranges[t * 8 + i] = uwbBatch[t].ranges(i);
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

        solver.solve(Uopt, xn, x0, U0, ranges, accgyr);
        cudaDeviceSynchronize();

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
        // Update initial state for next iteration
        std::copy(xn, xn + 15, x0);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "solve() time: "
                << std::chrono::duration<double, std::milli>(end - start).count()
                << " ms\n";
                
    poseFile.close();

    // int ret = std::system("python3 ../scripts/plot.py"); (void)ret;


    return 0;
}
