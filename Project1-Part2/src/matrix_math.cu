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

// ==============================================
// Kernel wrappers
// ==============================================

__global__ void kernMatAdd(float *a, float *b, float *c) {
    int i = (threadIdx.y * blockDim.x) + threadIdx.x;
    if (i < ARRAY_SIZE*ARRAY_SIZE) {
        c[i] = a[i] + b[i];
    }
}

__global__ void kernMatSub(float *a, float *b, float *c) {
    int i = (threadIdx.y * blockDim.x) + threadIdx.x;
    if (i < ARRAY_SIZE*ARRAY_SIZE) {
        c[i] = a[i] - b[i];
    }
}

__global__ void kernMatMul(float *a, float *b, float *c) {
    int index = (threadIdx.y * blockDim.x) + threadIdx.x;
    if (threadIdx.x >= ARRAY_SIZE || threadIdx.y >= ARRAY_SIZE) { return; }
    int sum = 0;
    for (int k = 0; k < ARRAY_SIZE; k++) {
        int i = (threadIdx.x * ARRAY_SIZE) + k;
        int j = (k * ARRAY_SIZE) + threadIdx.y;
        sum += a[i] + b[j];
    }
    c[index] = sum;
}

// ==============================================
// Kernel wrappers
// ==============================================

dim3 gridSize(1);
dim3 blockSize(ARRAY_SIZE, ARRAY_SIZE);

__host__ void cudaAdd(float *a, float *b, float *c) {
    cudaMemcpy(devA, a, ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
    kernMatAdd<<<gridSize, blockSize>>>(devA, devB, devC);
    cudaMemcpy(c, devC, ARRAY_MEM_SIZE, cudaMemcpyDeviceToHost);
}

__host__ void cudaSub(float *a, float *b, float *c) {
    cudaMemcpy(devA, a, ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
    kernMatSub<<<gridSize, blockSize>>>(devA, devB, devC);
    cudaMemcpy(c, devC, ARRAY_MEM_SIZE, cudaMemcpyDeviceToHost);
}

__host__ void cudaMul(float *a, float *b, float *c) {
    cudaMemcpy(devA, a, ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, b, ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
    kernMatMul<<<gridSize, blockSize>>>(devA, devB, devC);
    cudaMemcpy(c, devC, ARRAY_MEM_SIZE, cudaMemcpyDeviceToHost);
}
