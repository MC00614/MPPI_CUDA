// @file mppi_solver.cu
// @brief Implementation of the MPPI solver class.
// @author MC00614

#include "mppi_solver.cuh"

#include "drone_dynamics.cuh"

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
        c += 5.0 * uwb_step_cost(x_next, anchor, &ranges[t * 8]);
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
    cudaStreamCreate(&rngS_);
    cudaStreamCreate(&rollS_);
    cudaEventCreate(&rngDone_);

    // Init RNG
    cudaMalloc(&d_states, CHUNK * sizeof(curandStatePhilox4_32_10_t));
    init_rng<<<(CHUNK + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rngS_>>>(d_states, SEED, CHUNK);

    // Allocate device memory
    cudaMalloc(&d_noise, CHUNK * T_ * DIM_U * sizeof(double));
    cudaMalloc(&d_Ui, CHUNK * T_ * DIM_U * sizeof(double));
    cudaMalloc(&d_cost, CHUNK * sizeof(double));
}

MPPISolver::~MPPISolver() {
    cudaFree(d_states);
    cudaFree(d_anchor_);
    cudaFree(d_gravity_);
    cudaStreamDestroy(rngS_);
    cudaStreamDestroy(rollS_);
    cudaEventDestroy(rngDone_);

    cudaFree(d_noise);
    cudaFree(d_Ui);
    cudaFree(d_cost);
}

void MPPISolver::setDt(double dt) {
    dt_ = dt;
}

void MPPISolver::solve(double *h_Uopt, double *h_xn, const double *h_x0, const double *h_U0, const double *h_ranges, const double *h_accgyr) {
    const int len_U0 = T_ * DIM_U;
    const int len_ranges = T_ * 8;
    const int len_accgyr = T_ * 6;

    // Allocate device memory
    double* d_x0;
    double* d_U0;
    double* d_ranges;
    double* d_accgyr;
    cudaMalloc(&d_x0, DIM_X * sizeof(double));
    cudaMalloc(&d_U0, len_U0 * sizeof(double));
    cudaMalloc(&d_ranges, len_ranges * sizeof(double));
    cudaMalloc(&d_accgyr, len_accgyr * sizeof(double));
    cudaMemcpy(d_x0, h_x0, DIM_X * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_U0, h_U0, len_U0 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ranges, h_ranges, len_ranges * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_accgyr, h_accgyr, len_accgyr * sizeof(double), cudaMemcpyHostToDevice);
    
    // Host buffer
    if (h_cost_.size() < N_) {h_cost_.resize(N_);}
    if (h_Ui_.size() < N_ * T_ * DIM_U) {h_Ui_.resize(N_ * T_ * DIM_U);}

    int batch_start = 0;
    while (batch_start < N_) {
        int batch = std::min(CHUNK, N_ - batch_start); // Number of samples to process in this batch
        int batch_samples = batch * T_ * DIM_U; // Total number of input sample elements

        // Get noise
        noise_kernel<<<(batch_samples + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rngS_>>>(d_states, d_noise, batch, T_);
        cudaEventRecord(rngDone_, rngS_);

        // Wait for RNG to finish and then launch rollout
        cudaStreamWaitEvent(rollS_, rngDone_, 0);

        // Launch rollout kernel
        rollout_kernel<<<(batch + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, rollS_>>>(
            batch, T_, dt_, d_gravity_,
            d_anchor_, d_ranges, d_accgyr,
            d_U0, d_noise, d_x0, gamma_,
            d_Ui, d_cost);

        // Copy results back to host
        cudaMemcpyAsync(&h_cost_[batch_start], d_cost, batch * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
        cudaMemcpyAsync(&h_Ui_[batch_start * T_ * DIM_U], d_Ui, batch_samples * sizeof(double), cudaMemcpyDeviceToHost, rollS_);
        cudaStreamSynchronize(rollS_);

        batch_start += batch;
    }
    // Normalize costs and compute weighted average of control inputs
    double min_cost = *std::min_element(h_cost_.begin(), h_cost_.end());

    if (h_wi_.size() < N_) {h_wi_.resize(N_);}

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

    if (h_xn != nullptr) {
        // Compute final state using the average control input
        double* d_uopt;
        double* d_xn;
        cudaMalloc(&d_uopt, DIM_U * sizeof(double));
        cudaMalloc(&d_xn, DIM_X * sizeof(double));
        cudaMemcpy(d_uopt, h_Uopt, DIM_U * sizeof(double), cudaMemcpyHostToDevice);

        step_dynamics_kernel<<<1, 1>>>(d_x0, d_uopt, dt_, d_gravity_, d_xn);

        cudaMemcpy(h_xn, d_xn, DIM_X * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_uopt);
        cudaFree(d_xn);
    }

    // Free device memory
    cudaFree(d_x0);
    cudaFree(d_U0);
    cudaFree(d_ranges);
}
