// @file drone_dynamics.cuh
// @brief Header file for drone dynamics and control functions.
// @author MC00614

#pragma once
#include <cuda_runtime.h>
#include <cmath>

__device__ void vector_to_skew(const double* a, double* A_hat) {
    A_hat[0] =    0.0;   A_hat[1] = -a[2];   A_hat[2] =  a[1];
    A_hat[3] =  a[2];    A_hat[4] =   0.0;   A_hat[5] = -a[0];
    A_hat[6] = -a[1];    A_hat[7] =  a[0];   A_hat[8] =   0.0;
}

__device__ void Exp_SO3(const double* omega, double* R) {
    constexpr double TOL = 1e-6;
    double angle = sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2]);

    // Identity matrix
    R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
    R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
    R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;

    if (angle < TOL) return;

    double a[3] = {omega[0]/angle, omega[1]/angle, omega[2]/angle};
    double c = cos(angle);
    double s = sin(angle);

    double A_hat[9];
    vector_to_skew(a, A_hat);

    // Compute A_hat^2
    double A_hat2[9];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            A_hat2[3*i + j] = 0.0;
            for (int k = 0; k < 3; ++k) {
                A_hat2[3*i + j] += A_hat[3*i + k] * A_hat[3*k + j];
            }
        }
    }

    // R = I + s*A_hat + (1 - c)*A_hat^2
    for (int i = 0; i < 9; ++i) {
        R[i] += s * A_hat[i] + (1.0 - c) * A_hat2[i];
    }
}

__device__ inline double l2norm3(const double* a) {
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

__device__ void step_dynamics(const double* x,
                              const double* u,
                              double dt,
                              const double* g,
                              double* x_next) {
    double accWorld[3];
    accWorld[0] = x[6]*u[0] + x[7]*u[1] + x[8]*u[2] + g[0];
    accWorld[1] = x[9]*u[0] + x[10]*u[1] + x[11]*u[2] + g[1];
    accWorld[2] = x[12]*u[0] + x[13]*u[1] + x[14]*u[2] + g[2];
    x_next[0] = x[0] + x[3] * dt + 0.5*accWorld[0]*dt*dt;
    x_next[1] = x[1] + x[4] * dt + 0.5 * accWorld[1] * dt * dt;
    x_next[2] = x[2] + x[5] * dt + 0.5 * accWorld[2] * dt * dt;
    x_next[3] = x[3] + accWorld[0] * dt;
    x_next[4] = x[4] + accWorld[1] * dt;
    x_next[5] = x[5] + accWorld[2] * dt;

    double omega[3] = {u[3] * dt, u[4] * dt, u[5] * dt};
    double exp[9];
    Exp_SO3(omega, exp);
    x_next[6] = exp[0] * x[6] + exp[1] * x[7] + exp[2] * x[8];
    x_next[7] = exp[3] * x[6] + exp[4] * x[7] + exp[5] * x[8];
    x_next[8] = exp[6] * x[6] + exp[7] * x[7] + exp[8] * x[8];
    x_next[9] = exp[0] * x[9] + exp[1] * x[10] + exp[2] * x[11];
    x_next[10] = exp[3] * x[9] + exp[4] * x[10] + exp[5] * x[11];
    x_next[11] = exp[6] * x[9] + exp[7] * x[10] + exp[8] * x[11];
    x_next[12] = exp[0] * x[12] + exp[1] * x[13] + exp[2] * x[14];
    x_next[13] = exp[3] * x[12] + exp[4] * x[13] + exp[5] * x[14];
    x_next[14] = exp[6] * x[12] + exp[7] * x[13] + exp[8] * x[14];
}

__device__ double uwb_step_cost(const double* pos,
                                const double* anchor,
                                const double* ranges) {
    double tagL[3];
    for (int i = 0; i < 3; ++i) {
        tagL[i] = pos[i] + pos[7 + 3 * i] * 0.13;
    }
    double tagR[3];
    for (int i = 0; i < 3; ++i) {
        tagR[i] = pos[i] - pos[7 + 3 * i] * 0.13;
    }

    double Hx[16];
    for (int i = 0; i < 8; ++i) {
        double dL[3];
        dL[0] = tagL[0] - anchor[3*i + 0];
        dL[1] = tagL[1] - anchor[3*i + 1];
        dL[2] = tagL[2] - anchor[3*i + 2];
        double dR[3];
        dR[0] = tagR[0] - anchor[3*i + 0];
        dR[1] = tagR[1] - anchor[3*i + 1];
        dR[2] = tagR[2] - anchor[3*i + 2];

        Hx[i + 0] = sqrt(dL[0] * dL[0] + dL[1] * dL[1] + dL[2] * dL[2]);
        Hx[i + 8] = sqrt(dR[0] * dR[0] + dR[1] * dR[1] + dR[2] * dR[2]);
    }

    double s = 0.0;
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        double diff = Hx[j] - ranges[j];
        s += diff * diff;
    }
    return sqrt(s);
}


__device__ double imu_accgyr_step_cost(const double* pos, const double* imu) {
    double s = 0.0;
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        double diff = pos[j] - imu[j];
        s += diff * diff;
    }
    return sqrt(s);
}
