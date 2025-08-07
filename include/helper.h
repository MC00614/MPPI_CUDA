// @file main.cpp
// @brief Main program for the MPPI solver with IMU and UWB data processing.
// @author KYH7238

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <random>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

class STATE {
public:
    Eigen::Vector3d p;
    Eigen::Matrix3d R;
    Eigen::Vector3d v;
    STATE();
    STATE(const double* x) {
        p = Eigen::Vector3d(x[0], x[1], x[2]);
        v = Eigen::Vector3d(x[3], x[4], x[5]);
        R = Eigen::Map<const Eigen::Matrix3d>(x + 6);
    }
};

class ImuData {
public:
    double timeStamp;
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
};

class UwbData {
public:
    int id;                  // 0: left tag, 1: right tag (unused offline merged)
    double timeStamp;
    Eigen::VectorXd ranges;  // size = 8 for each tag (merged into 16 later)
};


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
