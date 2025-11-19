#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "queues/taskmanager_gpu.cuh"

int main() {
  printf("Host: entering main\n");

  const int N = 1024;

  double *dx = nullptr;
  double *dy = nullptr;
  cudaMalloc(&dx, N * sizeof(double));
  cudaMalloc(&dy, N * sizeof(double));
  printf("Host: allocated device memory\n");

  std::vector<double> hx(N, 1.0);
  cudaMemcpy(dx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  printf("Host: copied input to device\n");

  printf("Host: calling StartWorkersGPU (expectedTasks=1)\n");
  ASC_HPC::StartWorkersGPU(1, 128, 1);
  printf("Host: returned from StartWorkersGPU\n");

  ASC_HPC::GPU_Task task;
  task.type  = 0;
  task.n     = N;
  task.alpha = 2.0;
  task.x     = dx;
  task.y     = dy;

  printf("Host: calling EnqueueGPUTask\n");
  ASC_HPC::EnqueueGPUTask(task);
  printf("Host: returned from EnqueueGPUTask\n");

  printf("Host: calling WaitForAllGPU\n");
  ASC_HPC::WaitForAllGPU();
  printf("Host: returned from WaitForAllGPU\n");

  printf("Host: calling StopWorkersGPU\n");
  ASC_HPC::StopWorkersGPU();
  printf("Host: returned from StopWorkersGPU\n");

  std::vector<double> hy(N);
  cudaMemcpy(hy.data(), dy, N * sizeof(double), cudaMemcpyDeviceToHost);

  printf("Result check:\n");
  printf("  hy[0] = %f\n", hy[0]);
  printf("  hy[N-1] = %f\n", hy[N-1]);

  cudaFree(dx);
  cudaFree(dy);

  printf("Host: finished GPU queue test.\n");
  return 0;
}
