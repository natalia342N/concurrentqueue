#pragma once
#include <cuda_runtime.h>
#include "broker.cuh"

namespace ASC_HPC {

// -------------------------------
// GPU Task Structure
// -------------------------------

struct GPU_Task {
    int   type;    // 0 = y = alpha * x
    int   n;       // length of vectors
    double *x;     // device pointer
    double *y;     // device pointer
    double alpha;  // scalar
};

// Launches the single-kernel scheduler on the GPU.
// h_tasks: array of GPU_Task structs on the host
// numTasks: number of tasks
void RunSchedulerSingleKernel(GPU_Task* h_tasks, int numTasks);

} // namespace ASC_HPC
