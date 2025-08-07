// @file mppi_solver.cuh
// @brief Header file for the MPPI solver class.
// @author MC00614

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <vector>

#define BLOCK_SIZE 256  // threads per block
#define CHUNK 34816      // samples processed per batch
#define SEED 1234       // RNG seed

#define DIM_X 15
#define DIM_U 6

__constant__ double d_sigma_[6];

class MPPISolver {
public:
    MPPISolver(int N, int T, double dt, double gamma, const double* h_anchor, const double* h_gravity, const double* h_sigma);
    ~MPPISolver();
    void setDt(double dt);
    void solve(double *h_Uopt,
               double *xn,  // Storage for next state
               const double* h_x0,
               const double* h_U0,
               const double* h_ranges,
               const double* h_accgyr);
private:
    int N_;
    int T_;
    double dt_;
    double gamma_;
    double* d_anchor_{nullptr};
    double* d_gravity_{nullptr};

    cudaStream_t h2dS_{nullptr};
    cudaStream_t d2hS_{nullptr};
    cudaStream_t rngS_{nullptr};
    cudaStream_t rollS_{nullptr};
    cudaEvent_t  rngDone_{nullptr};

    std::vector<double> h_cost_;
    std::vector<double> h_Ui_;
    std::vector<double> h_wi_;

    curandStatePhilox4_32_10_t* d_states;

    int len_U0;
    int len_ranges;
    int len_accgyr;

    double* d_x0;
    double* d_U0;
    double* d_ranges;
    double* d_accgyr;

    double* d_noise;
    double* d_Ui;
    double* d_cost;
    double* d_wi;

    double* d_Uopt;
    
    double* d_xn;
};
