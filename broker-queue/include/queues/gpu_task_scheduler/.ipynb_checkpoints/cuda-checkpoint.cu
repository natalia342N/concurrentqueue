// daxpy_sin_dot_graph.cu  – build with:  nvcc -O3 -gencode arch=compute_86,code=sm_86 daxpy_sin_dot_graph.cu -o cuda_graph

#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <cmath>

#define CUDA_OK(c) do{cudaError_t e=(c); if(e!=cudaSuccess){            \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));\
  return EXIT_FAILURE; }}while(0)

constexpr int N = 1<<20;          // 1 048 576 doubles
const double  ALPHA = 2.0;

__global__ void daxpy(double a,const double* x,double* y,int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) y[i] = a*x[i] + y[i];
}

__global__ void apply_sin(double* y,int n){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n) y[i] = sin(y[i]);
}

__global__ void dot_kernel(const double* y,const double* z,double* out,int n){
    extern __shared__ double s[];
    int tid = threadIdx.x;
    int i   = blockIdx.x*blockDim.x + tid;
    s[tid]  = (i<n)? y[i]*z[i] : 0.0;
    __syncthreads();
    for(int stride=blockDim.x/2; stride>0; stride>>=1){
        if(tid<stride) s[tid]+=s[tid+stride];
        __syncthreads();
    }
    if(tid==0) atomicAdd(out,s[0]);        // double atomicAdd works in sm_60+
}

int main(){
    /* host data */
    std::vector<double> hx(N,1.0), hy(N,2.0), hz(N,3.0);
    double dot_ref = 0.0;

    /* device memory */
    double *dx,*dy,*dz,*dDot;
    CUDA_OK(cudaMalloc(&dx,N*sizeof(double)));
    CUDA_OK(cudaMalloc(&dy,N*sizeof(double)));
    CUDA_OK(cudaMalloc(&dz,N*sizeof(double)));
    CUDA_OK(cudaMalloc(&dDot,sizeof(double)));

    CUDA_OK(cudaMemcpy(dx,hx.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dy,hy.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dz,hz.data(),N*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dDot,0,sizeof(double)));

    int threads = 256;
    int blocks  = (N + threads - 1)/threads;

    cudaStream_t stream; CUDA_OK(cudaStreamCreate(&stream));
    cudaGraph_t g;       cudaGraphExec_t exe;

    /* ---------- capture ---------- */
    CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        daxpy<<<blocks,threads,0,stream>>>(ALPHA,dx,dy,N);
        apply_sin<<<blocks,threads,0,stream>>>(dy,N);
        dot_kernel<<<blocks,threads,threads*sizeof(double),stream>>>(dy,dz,dDot,N);
    CUDA_OK(cudaStreamEndCapture(stream,&g));
    CUDA_OK(cudaGraphInstantiate(&exe,g,nullptr,nullptr,0));

    /* ---------- launch ---------- */
    CUDA_OK(cudaGraphLaunch(exe,stream));
    CUDA_OK(cudaStreamSynchronize(stream));

    /* copy back */
    double dot_gpu;
    CUDA_OK(cudaMemcpy(&dot_gpu,dDot,sizeof(double),cudaMemcpyDeviceToHost));

    /* CPU reference */
    for(int i=0;i<N;i++){
        double y = ALPHA*hx[i] + hy[i];
        y        = sin(y);
        dot_ref += y*hz[i];
    }

    printf("dot_gpu = %.6e\n", dot_gpu);
    printf("dot_ref = %.6e\n", dot_ref);
    printf("abs diff = %.3e\n", fabs(dot_gpu-dot_ref));

    /* cleanup */
    cudaGraphExecDestroy(exe);
    cudaGraphDestroy(g);
    cudaStreamDestroy(stream);
    cudaFree(dx); cudaFree(dy); cudaFree(dz); cudaFree(dDot);
    return 0;
}
