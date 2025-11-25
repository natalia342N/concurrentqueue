#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sys/time.h>

constexpr int N = 1 << 20;
constexpr int REPEAT = 100;
const double ALPHA = 2.0;
const double BETA  = 3.0;

class Timer {
public:
  Timer() : ts(0) {}

  void reset() {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    ts = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
  }

  double get() const {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    double end_time = static_cast<double>(tval.tv_sec * 1000000 + tval.tv_usec);
    return (end_time - ts) / 1e6;
  }

private:
  double ts;  // ✅ This was missing!
};

__global__ void daxpy(double a, const double *x, double *y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

int main() {
  std::vector<double> hx(N, 1.0), hy(N, 2.0), hz(N, 3.0);
  std::vector<double> hy_ref = hy, hz_ref = hz;

  double *dx, *dy, *dz;
  cudaMalloc(&dx, N * sizeof(double));
  cudaMalloc(&dy, N * sizeof(double));
  cudaMalloc(&dz, N * sizeof(double));

  cudaMemcpy(dx, hx.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dy, hy.data(), N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(dz, hz.data(), N * sizeof(double), cudaMemcpyHostToDevice);

  int threads = 256;
  int blocks = (N + threads - 1) / threads;

  // Graph Capture Setup
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaGraph_t g;
  cudaGraphExec_t exec;

  cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
  daxpy<<<blocks, threads, 0, s>>>(ALPHA, dx, dy, N);
  daxpy<<<blocks, threads, 0, s>>>(BETA, dy, dz, N);
  cudaStreamEndCapture(s, &g);
  cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);

  // Warm-up
  cudaGraphLaunch(exec, s);
  cudaStreamSynchronize(s);

  // Timing
  Timer timer;
  std::vector<double> timings;
  for (int i = 0; i < REPEAT; ++i) {
    cudaDeviceSynchronize();
    timer.reset();
    cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);
    timings.push_back(timer.get());
  }

  std::sort(timings.begin(), timings.end());
  double median_time = timings[REPEAT / 2];
  double bandwidth_gb = (3.0 * N * sizeof(double)) / (median_time * 1e9); // 2 reads + 1 write per DAXPY × 2 ops

  std::cout << "[Graph Capture] Median Time = " << median_time << " s, Bandwidth = "
            << bandwidth_gb << " GB/s\n";

  // Copy results for validation
  cudaMemcpy(hy.data(), dy, N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(hz.data(), dz, N * sizeof(double), cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    hy_ref[i] = ALPHA * hx[i] + hy_ref[i];
    hz_ref[i] = BETA  * hy_ref[i] + hz_ref[i];
  }

  double max_err_y = 0.0, max_err_z = 0.0;
  for (int i = 0; i < N; i++) {
    max_err_y = fmax(max_err_y, fabs(hy[i] - hy_ref[i]));
    max_err_z = fmax(max_err_z, fabs(hz[i] - hz_ref[i]));
  }

  printf("Max |y_dev - y_ref| = %.3e\n", max_err_y);
  printf("Max |z_dev - z_ref| = %.3e\n", max_err_z);

  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(g);
  cudaStreamDestroy(s);
  cudaFree(dx); cudaFree(dy); cudaFree(dz);

  return 0;
}
