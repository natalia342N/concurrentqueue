#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "queues/taskmanager_gpu.cuh"

int main() {
  printf("Host: entering main\n");

  const int N = 1024;

  // Allocate device vectors
  double *dx = nullptr;
  double *dy = nullptr;
  cudaMalloc(&dx, N * sizeof(double));
  cudaMalloc(&dy, N * sizeof(double));
  printf("Host: allocated device memory\n");

  // Initialize x on host and copy to device
  std::vector<double> hx(N, 1.0);
  cudaMemcpy(dx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  printf("Host: copied input to device\n");

  // Prepare a single GPU_Task
  ASC_HPC::GPU_Task task;
  task.type  = 0;      // y = alpha * x
  task.n     = N;
  task.alpha = 2.0;
  task.x     = dx;     // device pointer
  task.y     = dy;     // device pointer

  // Launch the single-kernel scheduler with 1 task
  ASC_HPC::GPU_Task tasks[1] = { task };
  printf("Host: calling RunSchedulerSingleKernel\n");
  ASC_HPC::RunSchedulerSingleKernel(tasks, 1);
  printf("Host: returned from RunSchedulerSingleKernel\n");

  // Copy result back and check
  std::vector<double> hy(N);
  cudaMemcpy(hy.data(), dy, N * sizeof(double), cudaMemcpyDeviceToHost);

  printf("Result check:\n");
  printf("  hy[0]   = %f\n", hy[0]);
  printf("  hy[N-1] = %f\n", hy[N-1]);

  cudaFree(dx);
  cudaFree(dy);

  printf("Host: finished GPU BrokerQueue single-kernel test.\n");
  return 0;
}
