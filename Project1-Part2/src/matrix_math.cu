#include "matrix_math.h"


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

int N = 25;
float* dev_mat_a;
float* dev_mat_b;
float* dev_mat_c;

/**
 * Initialize memory, update some globals
 */
void Matrix_Math::initialize() {
    int N = 25;
    dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

    cudaMalloc((void**)&hst_mat, N * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc hst_mat failed!");

    cudaMalloc((void**)&dev_mat_a, N * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_mat_a failed!");

    cudaMalloc((void**)&dev_mat_b, N * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_mat_b failed!");

    cudaMalloc((void**)&dev_mat_c, N * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_mat_c failed!");

    cudaThreadSynchronize();
}

void Matrix_Math::cleanUp() {
    cudaFree(dev_mat_a);
    cudaFree(dev_mat_b);
    cudaFree(dev_mat_c);
}

__global__ void mat_add(float *A, float *B, float *C) {
}

__global__ void mat_sub(float *A, float *B, float *C) {
}

__global__ void mat_mul(float *A, float *B, float *C) {
}