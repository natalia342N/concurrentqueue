#include <cuda_runtime.h>
#include <stdio.h>
#include <chrono>

#define N 500000           // number of elements per kernel
#define NSTEP 1000         // number of timesteps
#define NKERNEL 20         // number of kernels per timestep

__global__ void shortKernel(float *out_d, const float *in_d) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out_d[idx] = 1.23f * in_d[idx];
}

double elapsed_ms(std::chrono::high_resolution_clock::time_point t0,
                  std::chrono::high_resolution_clock::time_point t1) {
    return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
}

int main() {
    const int bytes = N * sizeof(float);
    float *in_d, *out_d;
    cudaMalloc(&in_d, bytes);
    cudaMalloc(&out_d, bytes);

    int threads = 512;
    int blocks  = (N + threads - 1) / threads;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // 1. Naive launch with sync after each kernel
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int istep = 0; istep < NSTEP; istep++) {
        for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
            cudaStreamSynchronize(stream);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("[Naive] Total time: %.3f ms, per-kernel: %.3f us\n",
           elapsed_ms(t0, t1),
           elapsed_ms(t0, t1) * 1000.0 / (NSTEP * NKERNEL));

    // 2. Launches without per-kernel sync (one sync per timestep)
    t0 = std::chrono::high_resolution_clock::now();
    for (int istep = 0; istep < NSTEP; istep++) {
        for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
            shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
        }
        cudaStreamSynchronize(stream);
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("[Overlap] Total time: %.3f ms, per-kernel: %.3f us\n",
           elapsed_ms(t0, t1),
           elapsed_ms(t0, t1) * 1000.0 / (NSTEP * NKERNEL));

    // 3. Using CUDA Graph
    bool graphCreated = false;
    cudaGraph_t graph;
    cudaGraphExec_t instance;

    t0 = std::chrono::high_resolution_clock::now();
    for (int istep = 0; istep < NSTEP; istep++) {
        if (!graphCreated) {
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
            for (int ikrnl = 0; ikrnl < NKERNEL; ikrnl++) {
                shortKernel<<<blocks, threads, 0, stream>>>(out_d, in_d);
            }
            cudaStreamEndCapture(stream, &graph);
            cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
            graphCreated = true;
        }
        cudaGraphLaunch(instance, stream);
        cudaStreamSynchronize(stream);
    }
    t1 = std::chrono::high_resolution_clock::now();
    printf("[Graph] Total time: %.3f ms, per-kernel: %.3f us\n",
           elapsed_ms(t0, t1),
           elapsed_ms(t0, t1) * 1000.0 / (NSTEP * NKERNEL));

    cudaGraphExecDestroy(instance);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(in_d);
    cudaFree(out_d);
    return 0;
}
