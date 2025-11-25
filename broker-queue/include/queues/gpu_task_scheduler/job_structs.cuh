#pragma once

struct Job {
    void (*func)(void*, int, int);
    void* data;
    int n;
    int* completed;
};

// 🔁 Declare only — don't define
extern __device__ int current_phase;
extern __device__ Job device_job;
extern __device__ Job device_job2;

struct Task {
    Job* job;
    int i;

    __device__ void execute() const {
        int phase = atomicAdd(&current_phase, 0);
        int job_id = (job == &device_job) ? 0 : 1;

        if (job_id != phase) return;

        printf("Executing task %d from job %d\n", i, job_id);
        job->func(job->data, i, job->n);

        int done = atomicAdd(job->completed, 1) + 1;
        if (done == job->n) {
            printf("All %d tasks complete. Job %d done.\n", job->n, job_id);
            if (job_id == 0) {
                atomicExch(&current_phase, 1);
            }
        }
    }
};
