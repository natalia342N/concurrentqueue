#pragma once
#include "job_structs.cuh"
#include "broker.cuh"

// === Device Global Queue ===
__device__ BrokerQueue<1024, Task, 10000> globalQueue;

// === Queue Initialization Kernel ===
__global__ void init_queue() {
    globalQueue.init();
    printf("Queue initialized.\n");
}

// === Task Processing Kernel with Phase Control ===
__global__ void process_tasks_kernel() {
    int tid = threadIdx.x;

    while (true) {
        Task task;
        bool hasTask;
        globalQueue.dequeue(hasTask, task);

        if (!hasTask) return;

        int phase = atomicAdd(&current_phase, 0);
        int job_id = (task.job == &device_job) ? 0 : 1;

        if (job_id != phase) {
            globalQueue.enqueue(task);
            continue;
        }

        printf("Thread %d dequeued task %d\n", tid, task.i);
        task.execute();

        __syncthreads();
    }
}