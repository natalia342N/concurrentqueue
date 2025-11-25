#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

// Kernel 1: Vector updates (x, r, p) and compute partial <r, r>
__global__ void kernel_update_vectors(int N, double *x, double *p, double *r, double *Api,
                                      double alpha, double beta, double *partial_rr_dot) {
    __shared__ double shared_rr_dot[512]; // Fixed size for shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double local_rr_dot = 0.0;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        x[i] += alpha * p[i];
        r[i] -= alpha * Api[i];
        p[i] = r[i] + beta * p[i];
        local_rr_dot += r[i] * r[i];
    }

    // Reduction in shared memory
    shared_rr_dot[threadIdx.x] = local_rr_dot;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_rr_dot[threadIdx.x] += shared_rr_dot[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_rr_dot[blockIdx.x] = shared_rr_dot[0];
    }
}

// Kernel 2: Compute Api, <Api, Api>, and <p, Api>
__global__ void kernel_compute_Api_and_dots(int N, int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                                            double *p, double *Api, double *partial_Api_dot, double *partial_pApi_dot) {
    __shared__ double shared_Api_dot[512];
    __shared__ double shared_pApi_dot[512];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double local_Api_dot = 0.0;
    double local_pApi_dot = 0.0;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
            sum += csr_values[k] * p[csr_colindices[k]];
        }
        Api[i] = sum;
        local_Api_dot += sum * sum;
        local_pApi_dot += p[i] * sum;
    }

    // Reduction in shared memory
    shared_Api_dot[threadIdx.x] = local_Api_dot;
    shared_pApi_dot[threadIdx.x] = local_pApi_dot;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (threadIdx.x < offset) {
            shared_Api_dot[threadIdx.x] += shared_Api_dot[threadIdx.x + offset];
            shared_pApi_dot[threadIdx.x] += shared_pApi_dot[threadIdx.x + offset];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial_Api_dot[blockIdx.x] = shared_Api_dot[0];
        partial_pApi_dot[blockIdx.x] = shared_pApi_dot[0];
    }
}

// Helper function for device reduction
double reduce_partial_results(double *partial_results, int num_blocks) {
    double result = 0.0;
    std::vector<double> host_partial_results(num_blocks);
    cudaMemcpy(host_partial_results.data(), partial_results, sizeof(double) * num_blocks, cudaMemcpyDeviceToHost);
    for (double val : host_partial_results) {
        result += val;
    }
    return result;
}

// Conjugate Gradient Solver
void conjugate_gradient(int N, int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution) {
    Timer timer;
    std::fill(solution, solution + N, 0);

    double alpha, beta, initial_residual_squared, residual_norm_squared;

    double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap;
    double *partial_rr_dot, *partial_Api_dot, *partial_pApi_dot;

    int num_threads = 512;
    int num_blocks = (N + num_threads - 1) / num_threads;
    cudaMalloc(&cuda_solution, sizeof(double) * N);
    cudaMalloc(&cuda_p, sizeof(double) * N);
    cudaMalloc(&cuda_r, sizeof(double) * N);
    cudaMalloc(&cuda_Ap, sizeof(double) * N);
    cudaMalloc(&partial_rr_dot, sizeof(double) * num_blocks);
    cudaMalloc(&partial_Api_dot, sizeof(double) * num_blocks);
    cudaMalloc(&partial_pApi_dot, sizeof(double) * num_blocks);

    cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);

    kernel_update_vectors<<<num_blocks, num_threads>>>(
        N, cuda_solution, cuda_p, cuda_r, cuda_Ap, 0.0, 0.0, partial_rr_dot);
    cudaDeviceSynchronize();
    initial_residual_squared = reduce_partial_results(partial_rr_dot, num_blocks);
    residual_norm_squared = initial_residual_squared;

    int iters = 0;
    timer.reset();
    while (true) {
        kernel_compute_Api_and_dots<<<num_blocks, num_threads>>>(
            N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap, partial_Api_dot, partial_pApi_dot);
        cudaDeviceSynchronize();

        double Api_dot = reduce_partial_results(partial_Api_dot, num_blocks);
        double pApi_dot = reduce_partial_results(partial_pApi_dot, num_blocks);
        alpha = residual_norm_squared / pApi_dot;

        kernel_update_vectors<<<num_blocks, num_threads>>>(
            N, cuda_solution, cuda_p, cuda_r, cuda_Ap, alpha, beta, partial_rr_dot);
        cudaDeviceSynchronize();

        double new_residual_norm_squared = reduce_partial_results(partial_rr_dot, num_blocks);
        if (std::sqrt(new_residual_norm_squared / initial_residual_squared) < 1e-6) {
            break;
        }

        beta = new_residual_norm_squared / residual_norm_squared;
        residual_norm_squared = new_residual_norm_squared;

        if (++iters > 10000) {
            std::cout << "Conjugate Gradient did NOT converge within 10000 iterations" << std::endl;
            break;
        }
    }

    cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    std::cout << "Time per iteration: " << timer.get() / iters << " seconds" << std::endl;

    cudaFree(cuda_solution);
    cudaFree(cuda_p);
    cudaFree(cuda_r);
    cudaFree(cuda_Ap);
    cudaFree(partial_rr_dot);
    cudaFree(partial_Api_dot);
    cudaFree(partial_pApi_dot);
}

// Solve System
void solve_system(int points_per_direction) {
    int N = points_per_direction * points_per_direction;

    std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

    int *csr_rowoffsets = (int *)malloc(sizeof(int) * (N + 1));
    int *csr_colindices = (int *)malloc(sizeof(int) * 5 * N);
    double *csr_values = (double *)malloc(sizeof(double) * 5 * N);

    double *solution = (double *)malloc(sizeof(double) * N);
    double *rhs = (double *)malloc(sizeof(double) * N);
    std::fill(rhs, rhs + N, 1);

    int *cuda_csr_rowoffsets, *cuda_csr_colindices;
    double *cuda_csr_values;
    cudaMalloc(&cuda_csr_rowoffsets, sizeof(int) * (N + 1));
    cudaMalloc(&cuda_csr_colindices, sizeof(int) * 5 * N);
    cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);

    generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

    cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(int) * 5 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_csr_values, csr_values, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);

    conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);

    cudaFree(cuda_csr_rowoffsets);
    cudaFree(cuda_csr_colindices);
    cudaFree(cuda_csr_values);
    free(solution);
    free(rhs);
    free(csr_rowoffsets);
    free(csr_colindices);
    free(csr_values);
}

int main() {
    std::vector<int> grid_sizes = {1000};

    for (int size : grid_sizes) {
        solve_system(size);
    }

    return EXIT_SUCCESS;
}
