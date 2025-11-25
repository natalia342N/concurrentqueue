// include/queues/taskmanager_gpu.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "queues/taskmanager_gpu.cuh"

namespace ASC_HPC {

  // --------------------------------
  // small helper for error checking
  // --------------------------------
  inline void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
      printf("CUDA ERROR in %s: %s (%d)\n",
             what, cudaGetErrorString(err), (int)err);
      fflush(stdout);
      std::abort();
    }
  }

  // --------------------------------
  // device-side globals
  // --------------------------------

  __device__ BrokerQueue<1024, GPU_Task, 100000> globalQueue;
  __device__ int d_doneCounter = 0;  // how many tasks have been completed

  // --------------------------------
  // Single-kernel scheduler
  //
  // 1) Initializes the queue
  // 2) Enqueues all tasks (by thread 0)
  // 3) All threads cooperatively dequeue + execute tasks
  // --------------------------------

  __global__ void scheduler_kernel(GPU_Task* tasks, int numTasks) {
    // 1) init queue + doneCounter
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      globalQueue.init();
      d_doneCounter = 0;
      printf("scheduler_kernel: queue initialized, numTasks=%d\n", numTasks);
    }
    __syncthreads();

    // 2) producer phase: single producer enqueues all tasks
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      for (int i = 0; i < numTasks; ++i) {
        bool ok = globalQueue.enqueue(tasks[i]);
        printf("scheduler_kernel: enqueue task %d, ok=%d, type=%d, n=%d\n",
               i, (int)ok, tasks[i].type, tasks[i].n);
      }
    }
    __syncthreads();

    // 3) worker phase: all threads repeatedly dequeue + execute
    const int maxIters = 1000000;
    int iter = 0;

    while (iter < maxIters) {
      int done = d_doneCounter;
      if (done >= numTasks) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("scheduler_kernel: all tasks done (done=%d)\n", done);
        }
        break;
      }

      bool     hasTask = false;
      GPU_Task task;

      globalQueue.dequeue(hasTask, task);

      if (hasTask) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("scheduler_kernel: GOT task, type=%d, n=%d\n",
                 task.type, task.n);
        }

        // execute task type 0: y[i] = alpha * x[i]
        if (task.type == 0) {
          for (int i = threadIdx.x; i < task.n; i += blockDim.x) {
            task.y[i] = task.alpha * task.x[i];
          }
        }

        __syncthreads();
        if (threadIdx.x == 0) {
          atomicAdd(&d_doneCounter, 1);
        }
      } else {
        // no work right now, back off a bit
        if (threadIdx.x == 0 && blockIdx.x == 0 && (iter % 200000) == 0) {
          printf("scheduler_kernel: no task at iter=%d, done=%d\n",
                 iter, d_doneCounter);
        }
        __nanosleep(50);
      }

      ++iter;
      __syncthreads();
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("scheduler_kernel: EXIT, final iter=%d, done=%d\n",
             iter, d_doneCounter);
    }
  }

  // --------------------------------
  // Host wrapper
  // --------------------------------

  void RunSchedulerSingleKernel(GPU_Task* h_tasks, int numTasks) {
    printf("Host: RunSchedulerSingleKernel - numTasks=%d\n", numTasks);

    // copy tasks to device (note: x,y pointers inside must already be device pointers)
    GPU_Task* d_tasks = nullptr;
    checkCuda(cudaMalloc(&d_tasks, numTasks * sizeof(GPU_Task)),
              "cudaMalloc(d_tasks)");
    checkCuda(cudaMemcpy(d_tasks, h_tasks, numTasks * sizeof(GPU_Task),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy(d_tasks)");

    // launch scheduler kernel: 1 block, 128 threads for now
    dim3 blocks(1);
    dim3 threads(128);
    scheduler_kernel<<<blocks, threads>>>(d_tasks, numTasks);
    checkCuda(cudaGetLastError(), "kernel launch (scheduler_kernel)");

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (scheduler_kernel)");

    cudaFree(d_tasks);
  }

} // namespace ASC_HPC
