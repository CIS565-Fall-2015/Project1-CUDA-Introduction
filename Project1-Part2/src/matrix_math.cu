#include <stdio.h>
#include <cuda.h>
#include <cmath>

#include "config.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAError(const char *msg, int line = -1) {
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
        if (line >= 0) {
            fprintf(stderr, "Line %d: ", line);
        }
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ==============================================
// Initialization
// ==============================================

float *devA;
float *devB;
float *devC;

__host__ int dev_init() {
    if (int i = cudaMalloc((void**)&devA, ARRAY_MEM_SIZE)) {
        return i;
    }
    if (int i = cudaMalloc((void**)&devB, ARRAY_MEM_SIZE)) {
        cudaFree(devA);
        return i;
    }
    if (int i = cudaMalloc((void**)&devC, ARRAY_MEM_SIZE)) {
        cudaFree(devA);
        cudaFree(devB);
        return i;
    }
    return 0;
}

__host__ void dev_free() {
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
}

__host__ void hst_init(float **a, float **b, float **c) {
    *a = (float*)malloc(ARRAY_MEM_SIZE);
    *b = (float*)malloc(ARRAY_MEM_SIZE);
    *c = (float*)malloc(ARRAY_MEM_SIZE);
}

__host__ void hst_free(float *a, float *b, float *c) {
    free(a);
    free(b);
    free(c);
}
