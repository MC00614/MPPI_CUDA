// @file mppi_solver.cu
// @brief Implementation of the MPPI solver class.
// @author MC00614

#include "mppi_solver.cuh"

#include "drone_dynamics.cuh"

// #include <cfloat>
// #include <thrust/device_ptr.h>
// #include <thrust/transform.h>
// #include <thrust/reduce.h>
// #include <thrust/fill.h>
// #include <thrust/for_each.h>
// #include <thrust/execution_policy.h>
// #include <thrust/iterator/counting_iterator.h>
// #include <thrust/iterator/constant_iterator.h>

#include <algorithm>
#include <numeric>

#include <iostream>
__global__ void init_rng(curandStatePhilox4_32_10_t *states, unsigned long seed, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) curand_init(seed, i, i * 1024ULL, &states[i]);
}

__global__ void noise_kernel(curandStatePhilox4_32_10_t *states, double *noise, int batch, int T) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch * T) return;

    int sample = idx / T;
    int time = idx % T;
    curandStatePhilox4_32_10_t local = states[sample];

    int base = (sample * T + time) * DIM_U;
    #pragma unroll
    for (int j = 0; j < DIM_U; ++j) {
        noise[base + j] = curand_normal_double(&local) * d_sigma_[j];
    }

    states[sample] = local;
}

__global__ void rollout_kernel(int N, int T, double dt, const double* g,
                               const double* anchor, const double* ranges, const double* accgyr,
                               const double* U0, const double* noise, const double* x0,
                               double gamma,
                               double* Ui, double* cost) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= N) return;

    double x[DIM_X];
    #pragma unroll
    for (int i = 0; i < DIM_X; ++i) {
        x[i] = x0[i];
    }

    double c = 0.0;

    const double* noise_base = noise + sample * T * DIM_U;
    double* ui_base = Ui + sample * T * DIM_U;
    for (int t = 0; t < T; ++t) {
        const int t_dim_u = t * DIM_U;
        const double* n = noise_base + t_dim_u;
        const double* u0 = U0 + t_dim_u;
        double* ui = ui_base + t_dim_u;

        #pragma unroll
        for (int k = 0; k < DIM_U; ++k)
            ui[k] = u0[k] + n[k];

        double x_next[DIM_X];
        step_dynamics(x, ui, dt, g, x_next);
        // UWB cost
        c += 5.0 * uwb_step_cost(x_next, anchor, &ranges[t * 16]);
        // IMU ACC cost
        c += 1e-3 * imu_accgyr_step_cost(ui, &accgyr[t * 6]);
        // IMU GYR cost
        c += 1e-1 * imu_accgyr_step_cost(ui + 3, &accgyr[t * 6 + 3]);

        #pragma unroll
        for (int k = 0; k < DIM_X; ++k)
            x[k] = x_next[k];
    }

    cost[sample] = c;
}

__global__ void step_dynamics_kernel(const double* x, const double* u, double dt, const double* g, double* x_next) {
    step_dynamics(x, u, dt, g, x_next);
}


MPPISolver::MPPISolver(int N, int T, double dt, double gamma, const double *h_anchor, const double *h_gravity, const double *h_sigma)
    : N_(N), T_(T), dt_(dt), gamma_(gamma) {
    cudaMemcpyToSymbol(d_sigma_, h_sigma, DIM_U * sizeof(double));

    // Allocate device memory for anchor
    cudaMalloc(&d_anchor_, 24 * sizeof(double));
    cudaMemcpy(d_anchor_, h_anchor, 24 * sizeof(double), cudaMemcpyHostToDevice);

    // Allocate device memory for gravity
    cudaMalloc(&d_gravity_, 3 * sizeof(double));
    cudaMemcpy(d_gravity_, h_gravity, 3 * sizeof(double), cudaMemcpyHostToDevice);

    // Create CUDA streams and events
    cudaStreamCreate(&h2dS_);
    cudaStreamCreate(&d2hS_);
    // Create streams for RNG and rollout
    cudaStreamCreate(&rngS_);
    cudaStreamCreate(&rollS_);
    cudaEventCreate(&rngDone_);

    // Init RNG
    cudaMalloc(&d_states, CHUNK * sizeof(curandStatePhilox4_32_10_t));
    init_rng<<<(CHUNK + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rngS_>>>(d_states, SEED, CHUNK);

    // Resize host vectors
    h_cost_.resize(N_);
    h_Ui_.resize(N_ * T_ * DIM_U);
    h_wi_.resize(N_);

    // Allocate device memory
    len_U0 = T_ * DIM_U;
    len_ranges = T_ * 16;
    len_accgyr = T_ * 6;
    cudaMalloc(&d_x0, DIM_X * sizeof(double));
    cudaMalloc(&d_U0, len_U0 * sizeof(double));
    cudaMalloc(&d_ranges, len_ranges * sizeof(double));
    cudaMalloc(&d_accgyr, len_accgyr * sizeof(double));

    cudaMalloc(&d_noise, N_ * T_ * DIM_U * sizeof(double));
    cudaMalloc(&d_Ui, N_ * T_ * DIM_U * sizeof(double));
    cudaMalloc(&d_cost, N_ * sizeof(double));
    cudaMalloc(&d_wi, N_ * sizeof(double));
    cudaMalloc(&d_Uopt, T_ * DIM_U * sizeof(double));

    cudaMalloc(&d_xn, DIM_X * sizeof(double));
}

MPPISolver::~MPPISolver() {
    cudaFree(d_states);
    cudaFree(d_anchor_);
    cudaFree(d_gravity_);
    cudaStreamDestroy(h2dS_);
    cudaStreamDestroy(d2hS_);
    cudaStreamDestroy(rngS_);
    cudaStreamDestroy(rollS_);
    cudaEventDestroy(rngDone_);

    cudaFree(d_x0);
    cudaFree(d_U0);
    cudaFree(d_ranges);
    cudaFree(d_accgyr);

    cudaFree(d_noise);
    cudaFree(d_Ui);
    cudaFree(d_cost);
    cudaFree(d_wi);
    cudaFree(d_Uopt);

    cudaFree(d_xn);
}

void MPPISolver::setDt(double dt) {
    dt_ = dt;
}

void MPPISolver::solve(double *h_Uopt, double *h_xn, const double *h_x0, const double *h_U0, const double *h_ranges, const double *h_accgyr) {

    // Copy initial state and inputs to device
    cudaMemcpyAsync(d_x0, h_x0, DIM_X * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_U0, h_U0, len_U0 * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_ranges, h_ranges, len_ranges * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaMemcpyAsync(d_accgyr, h_accgyr, len_accgyr * sizeof(double), cudaMemcpyHostToDevice, h2dS_);
    cudaStreamSynchronize(h2dS_);

    int batch_start = 0;
    while (batch_start < N_) {
        int batch = std::min(CHUNK, N_ - batch_start); // Number of samples to process in this batch
        int batch_samples = batch * T_ * DIM_U; // Total number of input sample elements

        // Get noise
        noise_kernel<<<(batch_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rngS_>>>(d_states, d_noise + batch_start, batch, T_);
        cudaEventRecord(rngDone_, rngS_);

        // Wait for RNG to finish and then launch rollout
        cudaStreamWaitEvent(rollS_, rngDone_, 0);

        // Launch rollout kernel
        rollout_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rollS_>>>(
            batch, T_, dt_, d_gravity_,
            d_anchor_, d_ranges, d_accgyr,
            d_U0, d_noise, d_x0, gamma_,
            d_Ui + batch_start * T_ * DIM_U, d_cost + batch_start);

        batch_start += batch;
    }
    // // Copy results back to host
    cudaMemcpyAsync(h_cost_.data(), d_cost, N_ * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaMemcpyAsync(h_Ui_.data(), d_Ui, N_ * T_ * DIM_U * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
    cudaStreamSynchronize(rollS_);

    // auto policy = thrust::cuda::par.on(rollS_);
    // thrust::device_ptr<double> dCost(d_cost);
    // const double minCost = thrust::reduce(policy, dCost, dCost + N_, DBL_MAX, thrust::minimum<double>());

    // thrust::device_ptr<double> dWi(d_wi);
    // thrust::transform(policy, dCost, dCost + N_, dWi,
    //     [g = gamma_, m = minCost] __device__ (double c) {return exp(-g * (c - m));}
    // );

    // const double sumW = thrust::reduce(policy, dWi, dWi + N_);
    // thrust::transform(policy, dWi, dWi + N_, thrust::make_constant_iterator(sumW), dWi, thrust::divides<double>());

    // thrust::device_ptr<double> dUi(d_Ui);
    // thrust::device_ptr<double> dUopt(d_Uopt);
    
    // thrust::fill(policy, dUopt, dUopt + T_ * DIM_U, 0.0);


    // thrust::for_each_n(policy, thrust::make_counting_iterator<int>(0), N_,
    //     [T = T_, m = DIM_U, dUi = d_Ui, dWi = d_wi, dUo = d_Uopt] __device__ (int i) {
    //         const double w  = dWi[i];
    //         const double* in = dUi + i * T * m;
    //         for (int j = 0; j < T * m; ++j) {
    //             atomicAdd(dUo + j, w * in[j]);
    //         }
    //     }
    // );

    // if (h_xn != nullptr) {
    //     // Compute final state using the average control input
    //     step_dynamics_kernel<<<1, 1>>>(d_x0, d_Uopt, dt_, d_gravity_, d_xn);
    //     cudaMemcpy(h_xn, d_xn, DIM_X * sizeof(double), cudaMemcpyDeviceToHost);
    // }

    // Normalize costs and compute weighted average of control inputs
    double min_cost = *std::min_element(h_cost_.begin(), h_cost_.end());

    // CPU-side normalization of weights
    // MC Comment: (TODO) Possibly we can use GPU parallel reduction!
    double sum_w = 0.0;
    for (int i = 0; i < N_; ++i) {
        sum_w += (h_wi_[i] = exp(-gamma_ * (h_cost_[i] - min_cost)));
    }
    for (double& w : h_wi_) {
        w /= sum_w;
    }

    // MC Comment: (TODO) Possibly we can use GPU MatMul!
    std::fill(h_Uopt, h_Uopt + T_ * DIM_U, 0.0);
    for (int i = 0; i < N_; ++i) {
        for (int t = 0; t < T_; ++t) {
            for (int k = 0; k < DIM_U; ++k) {
                h_Uopt[t * DIM_U + k] += h_wi_[i] * h_Ui_[(i * T_ + t) * DIM_U + k];
            }
        }
    }

    cudaMemcpy(d_Uopt, h_Uopt, T_ * DIM_U * sizeof(double), cudaMemcpyHostToDevice);
    if (h_xn != nullptr) {
        // Compute final state using the average control input
        step_dynamics_kernel<<<1, 1>>>(d_x0, d_Uopt, dt_, d_gravity_, d_xn);
        cudaMemcpy(h_xn, d_xn, DIM_X * sizeof(double), cudaMemcpyDeviceToHost);
    }
}
