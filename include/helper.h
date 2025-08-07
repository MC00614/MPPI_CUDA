// @file main.cpp
// @brief Main program for the MPPI solver with IMU and UWB data processing.
// @author KYH7238

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <vector>

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