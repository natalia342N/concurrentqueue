#include "poisson2d.hpp"      
#include "timer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel_update_vectors(
        int N,
        double *x, double *p, double *r, const double *Api,
        const double *alpha, const double *beta,
        double *partial_rr_dot)
{
    __shared__ double s_rr[512];
    int idx  = blockIdx.x * blockDim.x + threadIdx.x;
    double a = *alpha;           // scalar broadcast
    double b = *beta;

    double local = 0.0;
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        x[i] += a * p[i];
        r[i] -= a * Api[i];
        p[i]  = r[i] + b * p[i];
        local += r[i] * r[i];
    }
    s_rr[threadIdx.x] = local;
    __syncthreads();

    for (int off = blockDim.x/2; off; off >>= 1) {
        if (threadIdx.x < off)
            s_rr[threadIdx.x] += s_rr[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial_rr_dot[blockIdx.x] = s_rr[0];
}

__global__ void kernel_compute_Api_and_dots(
        int N,
        const int *rowofs, const int *colidx, const double *vals,
        const double *p, double *Api,
        double *partial_Api_dot, double *partial_pApi_dot)
{
    __shared__ double s_Api [512];
    __shared__ double s_pApi[512];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double localA  = 0.0;
    double localpA = 0.0;

    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        double sum = 0.0;
        for (int k = rowofs[i]; k < rowofs[i+1]; ++k)
            sum += vals[k] * p[colidx[k]];
        Api[i] = sum;
        localA  += sum * sum;
        localpA += p[i] * sum;
    }
    s_Api [threadIdx.x] = localA;
    s_pApi[threadIdx.x] = localpA;
    __syncthreads();

    for (int off = blockDim.x/2; off; off >>= 1) {
        if (threadIdx.x < off) {
            s_Api [threadIdx.x] += s_Api [threadIdx.x + off];
            s_pApi[threadIdx.x] += s_pApi[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_Api_dot [blockIdx.x] = s_Api [0];
        partial_pApi_dot[blockIdx.x] = s_pApi[0];
    }
}

static double reduce_partials(double *d_part, int nBlocks)
{
    std::vector<double> h(nBlocks);
    cudaMemcpy(h.data(), d_part, nBlocks*sizeof(double),
               cudaMemcpyDeviceToHost);
    double sum = 0.0;
    for (double v : h) sum += v;
    return sum;
}

void conjugate_gradient(int N,
                        int *d_rowofs, int *d_colidx, double *d_vals,
                        double *rhs, double *solution)
{
    int nt = 512, nb = (N + nt - 1) / nt;

    /* device buffers */
    double *dx,*dp,*dr,*dAp,*d_rr,*d_Api,*d_pApi,*d_alpha,*d_beta;
    cudaMalloc(&dx,      N*sizeof(double));
    cudaMalloc(&dp,      N*sizeof(double));
    cudaMalloc(&dr,      N*sizeof(double));
    cudaMalloc(&dAp,     N*sizeof(double));
    cudaMalloc(&d_rr,    nb*sizeof(double));
    cudaMalloc(&d_Api,   nb*sizeof(double));
    cudaMalloc(&d_pApi,  nb*sizeof(double));
    cudaMalloc(&d_alpha, sizeof(double));
    cudaMalloc(&d_beta,  sizeof(double));

    cudaMemcpy(dp, rhs, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dr, rhs, N*sizeof(double), cudaMemcpyHostToDevice);

    double alpha=0.0, beta=0.0;
    cudaMemcpy(d_alpha,&alpha,sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta ,&beta ,sizeof(double),cudaMemcpyHostToDevice);

    kernel_update_vectors<<<nb,nt>>>(N,dx,dp,dr,dAp,d_alpha,d_beta,d_rr);
    cudaDeviceSynchronize();
    double r0 = reduce_partials(d_rr,nb), r = r0;

    cudaStream_t s;           cudaStreamCreate(&s);
    cudaGraph_t g;            cudaGraphExec_t gExec;

    cudaStreamBeginCapture(s,cudaStreamCaptureModeGlobal);
    kernel_update_vectors<<<nb,nt,0,s>>>(N,dx,dp,dr,dAp,d_alpha,d_beta,d_rr);
    cudaStreamEndCapture(s,&g);
    cudaGraphInstantiate(&gExec,g,nullptr,nullptr,0);

    int it=0; Timer t; t.reset();
    while (true) {
        kernel_compute_Api_and_dots<<<nb,nt>>>(N,d_rowofs,d_colidx,d_vals,
                                               dp,dAp,d_Api,d_pApi);
        cudaDeviceSynchronize();
        double pAp  = reduce_partials(d_pApi, nb);
        alpha = r / pAp;

        cudaMemcpy(d_alpha,&alpha,sizeof(double),cudaMemcpyHostToDevice);
        cudaMemcpy(d_beta ,&beta ,sizeof(double),cudaMemcpyHostToDevice);

        cudaGraphLaunch(gExec,s);
        cudaStreamSynchronize(s);

        double rNew = reduce_partials(d_rr,nb);
        if (std::sqrt(rNew/r0) < 1e-6) break;

        beta = rNew / r;
        r    = rNew;
        if (++it>10000){printf("No convergence\n");break;}
    }
    cudaMemcpy(solution,dx,N*sizeof(double),cudaMemcpyDeviceToHost);
    printf("iters %d   sec/iter %g\n",it,t.get()/it);

    cudaGraphExecDestroy(gExec); cudaGraphDestroy(g); cudaStreamDestroy(s);
    cudaFree(dx); cudaFree(dp); cudaFree(dr); cudaFree(dAp);
    cudaFree(d_rr); cudaFree(d_Api); cudaFree(d_pApi);
    cudaFree(d_alpha); cudaFree(d_beta);
}

void solve_system(int n)
{
    int N = n*n;
    printf("Solving %d unknowns\n",N);

    std::vector<int>    h_rowofs(N+1);
    std::vector<int>    h_colidx(5*N);
    std::vector<double> h_vals  (5*N);
    generate_fdm_laplace(n,h_rowofs.data(),h_colidx.data(),h_vals.data());

    int *d_rowofs,*d_colidx; double *d_vals;
    cudaMalloc(&d_rowofs,(N+1)*sizeof(int));
    cudaMalloc(&d_colidx,(5*N)*sizeof(int));
    cudaMalloc(&d_vals  ,(5*N)*sizeof(double));
    cudaMemcpy(d_rowofs,h_rowofs.data(),(N+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_colidx,h_colidx.data(),(5*N)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vals  ,h_vals.data()  ,(5*N)*sizeof(double),cudaMemcpyHostToDevice);

    std::vector<double> rhs(N,1.0), sol(N);
    conjugate_gradient(N,d_rowofs,d_colidx,d_vals,rhs.data(),sol.data());

    cudaFree(d_rowofs); cudaFree(d_colidx); cudaFree(d_vals);
}

int main()
{
    solve_system(1000);   // 1000 × 1000 grid  (1 000 000 unknowns)
    return 0;
}
