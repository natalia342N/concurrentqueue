#include <cuda_runtime.h>
#include <vector>
#include <cstdio>
#include <cmath>

constexpr int N      = 1 << 20;
const double  ALPHA  = 2.0;
const double  BETA   = 3.0;

__global__ void daxpy(double a, const double* x, double* y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main()
{
    std::vector<double> hx(N, 1.0), hy(N, 2.0), hz(N, 3.0);
    std::vector<double> hy_ref = hy,  hz_ref = hz;

    double *dx, *dy, *dz;
    cudaMalloc(&dx, N * sizeof(double));
    cudaMalloc(&dy, N * sizeof(double));
    cudaMalloc(&dz, N * sizeof(double));

    cudaMemcpy(dx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    cudaGraph_t       graph;
    cudaGraphExec_t   exec;
    cudaGraphCreate(&graph, 0);

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    cudaKernelNodeParams pA{};
    void* argsA[] = { (void*)&ALPHA, (void*)&dx, (void*)&dy, (void*)&N };

    pA.func            = (void*)daxpy;
    pA.gridDim         = dim3(blocks);
    pA.blockDim        = dim3(threads);
    pA.sharedMemBytes  = 0;
    pA.kernelParams    = argsA;
    pA.extra           = nullptr;

    cudaGraphNode_t nodeA;
    cudaGraphAddKernelNode(&nodeA, graph, nullptr, 0, &pA);

    cudaKernelNodeParams pB{};
    void* argsB[] = { (void*)&BETA, (void*)&dy, (void*)&dz, (void*)&N };

    pB.func            = (void*)daxpy;
    pB.gridDim         = dim3(blocks);
    pB.blockDim        = dim3(threads);
    pB.sharedMemBytes  = 0;
    pB.kernelParams    = argsB;
    pB.extra           = nullptr;

    cudaGraphNode_t nodeB;
    cudaGraphAddKernelNode(&nodeB, graph, nullptr, 0, &pB);

    cudaGraphAddDependencies(graph, &nodeA, &nodeB, 1);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    cudaStream_t s;
    cudaStreamCreate(&s);

    cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);

    cudaMemcpy(hy.data(), dy, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hz.data(), dz, N * sizeof(double), cudaMemcpyDeviceToHost);

    double max_err_y = 0.0, max_err_z = 0.0;
    for (int i = 0; i < N; ++i) {
        hy_ref[i] = ALPHA * hx[i] + hy_ref[i];
        hz_ref[i] =  BETA * hy_ref[i] + hz_ref[i];

        max_err_y = fmax(max_err_y, fabs(hy[i] - hy_ref[i]));
        max_err_z = fmax(max_err_z, fabs(hz[i] - hz_ref[i]));
    }

    printf("Max |y_dev - y_ref| = %.3e\n", max_err_y);
    printf("Max |z_dev - z_ref| = %.3e\n", max_err_z);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    return 0;
}
