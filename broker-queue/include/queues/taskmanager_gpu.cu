// include/queues/taskmanager_gpu.cu
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "queues/taskmanager_gpu.cuh"

namespace ASC_HPC {

  // --------------------------------
  // host-side worker stream
  // --------------------------------
  static cudaStream_t g_worker_stream = nullptr;

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

  __device__ BrokerQueue<1024, GPU_Task, 10000> globalQueue;
  __device__ int d_totalTasks   = 0;  // how many tasks must be processed
  __device__ int d_doneCounter  = 0;  // how many tasks have been completed

  // --------------------------------
  // queue init kernel
  // --------------------------------

  __global__ void init_queue_kernel() {
    globalQueue.init();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("init_queue_kernel: queue initialized\n");
    }
  }

  // --------------------------------
  // enqueue kernel (runs briefly)
  // --------------------------------

  __global__ void enqueue_kernel(GPU_Task task) {
    bool ok = globalQueue.enqueue(task);
    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("enqueue_kernel: ok=%d, type=%d, n=%d\n",
             (int)ok, task.type, task.n);
    }
  }

  // --------------------------------
  // worker kernel (persistent-ish)
  //
  // runs in its own stream while enqueue kernels
  // are launched in the default stream.
  // --------------------------------

  __global__ void worker_kernel() {
    int iter = 0;
    const int maxIters = 1000000;   // safety cap so we never hard-hang

    while (iter < maxIters) {

      int total = d_totalTasks;
      int done  = d_doneCounter;

      // all tasks processed?
      if (done >= total && total > 0) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("worker_kernel: all tasks done (done=%d, total=%d)\n",
                 done, total);
        }
        break;
      }

      bool     hasTask = false;
      GPU_Task task;

      globalQueue.dequeue(hasTask, task);

      if (hasTask) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
          printf("worker_kernel: GOT task type=%d n=%d\n",
                 task.type, task.n);
        }

        // execute task cooperatively: type 0 = y = alpha * x
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
        // no work *right now*; back off a bit
        if (threadIdx.x == 0 && blockIdx.x == 0 && (iter % 200000) == 0) {
          printf("worker_kernel: no task at iter=%d (total=%d, done=%d)\n",
                 iter, total, done);
        }
        __nanosleep(50);
      }

      ++iter;
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      printf("worker_kernel: EXIT, final iter=%d, total=%d, done=%d\n",
             iter, d_totalTasks, d_doneCounter);
    }
  }

  // --------------------------------
  // Host API
  // --------------------------------

  void StartWorkersGPU(int blocks, int threadsPerBlock, int expectedTasks) {
    printf("Host: StartWorkersGPU - expectedTasks=%d\n", expectedTasks);

    int zero = 0;
    checkCuda(cudaMemcpyToSymbol(d_doneCounter, &zero, sizeof(int)),
              "cudaMemcpyToSymbol(d_doneCounter)");
    checkCuda(cudaMemcpyToSymbol(d_totalTasks,  &expectedTasks, sizeof(int)),
              "cudaMemcpyToSymbol(d_totalTasks)");

    // create worker stream if needed
    if (!g_worker_stream) {
      checkCuda(cudaStreamCreate(&g_worker_stream),
                "cudaStreamCreate(g_worker_stream)");
    }

    // initialize queue in worker stream
    init_queue_kernel<<<1, 1, 0, g_worker_stream>>>();
    checkCuda(cudaGetLastError(), "kernel launch (init_queue_kernel)");

    // make sure init is done before workers start dequeuing
    checkCuda(cudaStreamSynchronize(g_worker_stream),
              "cudaStreamSynchronize (after init)");

    // launch worker kernel in worker stream (persists until all tasks done)
    worker_kernel<<<blocks, threadsPerBlock, 0, g_worker_stream>>>();
    checkCuda(cudaGetLastError(), "kernel launch (worker_kernel)");
  }

  void EnqueueGPUTask(const GPU_Task& t) {
    printf("Host: EnqueueGPUTask - launching enqueue_kernel\n");
    // launch in default stream so it can overlap with worker stream
    enqueue_kernel<<<1, 1>>>(t);
    checkCuda(cudaGetLastError(), "kernel launch (enqueue_kernel)");

    // for now we sync here to keep behaviour simple & deterministic
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (after enqueue)");
  }

  void WaitForAllGPU() {
    printf("Host: WaitForAllGPU - synchronizing worker stream\n");
    if (g_worker_stream) {
      checkCuda(cudaStreamSynchronize(g_worker_stream),
                "cudaStreamSynchronize(g_worker_stream)");
    }
    printf("Host: WaitForAllGPU - worker finished\n");
  }

  void StopWorkersGPU() {
    printf("Host: StopWorkersGPU\n");
    if (g_worker_stream) {
      cudaStreamDestroy(g_worker_stream);
      g_worker_stream = nullptr;
    }
  }

} // namespace ASC_HPC
