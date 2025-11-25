#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <cmath>

constexpr int N = 1 << 20;       
const double ALPHA = 2.0;
const double  BETA = 3.0;

__global__ void daxpy(double  a,
                      const double *x,
                      double *y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
    std::vector<double> hx(N, 1.0), hy(N, 2.0), hz(N, 3.0);
    std::vector<double> hy_ref = hy, hz_ref = hz;

    double *dx, *dy, *dz;
    cudaMalloc(&dx, N*sizeof(double));
    cudaMalloc(&dy, N*sizeof(double));
    cudaMalloc(&dz, N*sizeof(double));

    cudaMemcpy(dx, hx.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dz, hz.data(), N*sizeof(double), cudaMemcpyHostToDevice);

    //graph capture
    int threads = 256;
    int blocks  = (N + threads - 1) / threads;

    cudaStream_t s;
    cudaStreamCreate(&s);

    cudaGraph_t g;
    cudaGraphExec_t exec;

    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);

    daxpy<<<blocks, threads, 0, s>>>(ALPHA, dx, dy, N);  // y = αx + y
    daxpy<<<blocks, threads, 0, s>>>( BETA, dy, dz, N);  // z = βy + z

    cudaStreamEndCapture(s, &g);
    cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);

    //execute graph
    cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);

    cudaMemcpy(hy.data(), dy, N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hz.data(), dz, N*sizeof(double), cudaMemcpyDeviceToHost);

    for (int i=0;i<N;i++){
        hy_ref[i] = ALPHA*hx[i] + hy_ref[i];
        hz_ref[i] =  BETA*hy_ref[i] + hz_ref[i];
    }

    double max_err_y = 0.0, max_err_z = 0.0;
    for (int i=0;i<N;i++){
        max_err_y = fmax(max_err_y, fabs(hy[i]-hy_ref[i]));
        max_err_z = fmax(max_err_z, fabs(hz[i]-hz_ref[i]));
    }

    printf("Max |y_dev - y_ref| = %.3e\n", max_err_y);
    printf("Max |z_dev - z_ref| = %.3e\n", max_err_z);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    return 0;
}
