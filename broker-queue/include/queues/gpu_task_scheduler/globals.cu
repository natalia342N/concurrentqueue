#include <cuda_runtime.h>
#include <cstdio>  // for printf if needed

// ✅ Just define everything manually here
struct Job {
    void (*func)(void*, int, int);
    void* data;
    int n;
    int* completed;
};

// Now define the globals
__device__ Job device_job;
__device__ Job device_job2;
__device__ int current_phase = 0;
