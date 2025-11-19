#pragma once
#include <cuda_runtime.h>
#include "broker.cuh"

namespace ASC_HPC {

// -------------------------------
// GPU Task Structure
// -------------------------------

struct GPU_Task {
    int   type;
    int   n;
    double *x;
    double *y;
    double alpha;
};

// -------------------------------
// Public API (host)
// -------------------------------

// expectedTasks: how many tasks will be enqueued in total
void StartWorkersGPU(int blocks, int threadsPerBlock, int expectedTasks);

// wait until all tasks are processed by the workers
void WaitForAllGPU();

// clean up worker stream (and in future: force-stop workers)
void StopWorkersGPU();

// enqueue a GPU task into the BrokerQueue on the device
void EnqueueGPUTask(const GPU_Task& t);

} // namespace ASC_HPC
