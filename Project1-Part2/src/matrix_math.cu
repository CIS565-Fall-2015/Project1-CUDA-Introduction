#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
//#include "utilityCore.hpp"
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


/*****************
 * Configuration *
 *****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 1024


/***********************************************
 * Kernel state  *
 ***********************************************/

dim3 threadsPerBlock(blockSize);
dim3 fullBlocksPerGrid((25 + blockSize - 1) / blockSize);

float *dev_A;
float *dev_B;
float *dev_C;


/******************
 * init *
 ******************/


/**
 * Initialize memory, update some globals
 */
void Matrix_Math::init() {
    //dim3 fullBlocksPerGrid(1);

    cudaMalloc((void**)&dev_A, 25 * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

    cudaMalloc((void**)&dev_B, 25 * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

    cudaMalloc((void**)&dev_C, 25 * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_C failed!");

//    cudaMemcpy(dev_A, hst_A, 25 * sizeof(float), cudaMemcpyHostToDevice);
//    checkCUDAErrorWithLine("cudaMemcpy hst_A to dev_A failed!");
//
//    cudaMemcpy(dev_B, hst_B, 25 * sizeof(float), cudaMemcpyHostToDevice);
//    checkCUDAErrorWithLine("cudaMemcpy hst_B to dev_B failed!");

    //cudaThreadSynchronize();
}

/******************
 * Matrix_Math *
 ******************/

__global__ void mat_add(float *dev_A, float *dev_B, float *dev_C) {
    // TODO: implement updateAccArray.
    // This function body runs once on each CUDA thread.
    // To avoid race conditions, each instance should only write ONE value to `acc`!
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	dev_C[index] = dev_A[index] + dev_B[index];
}

__global__ void mat_sub(float *dev_A, float *dev_B, float *dev_C) {
    // TODO: implement updateAccArray.
    // This function body runs once on each CUDA thread.
    // To avoid race conditions, each instance should only write ONE value to `acc`!
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	dev_C[index] = dev_A[index] - dev_B[index];
}

__global__ void mat_mul(float *dev_A, float *dev_B, float *dev_C) {
    // TODO: implement updateAccArray.
    // This function body runs once on each CUDA thread.
    // To avoid race conditions, each instance should only write ONE value to `acc`!
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	int row = index/5;
	int col = index%5;
	float result = 0;
	for (int i=0; i<5; i++){
		result = result + dev_A[(row*5) + i] * dev_B[(i*5) + col];
	}
	dev_C[index] = result;
}

/******************
 * Matrix_Math *
 ******************/
float Matrix_Math::add(float* A, float* B, float* C){
    init();
	cudaMemcpy(dev_A, A, 25 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, 25 * sizeof(float), cudaMemcpyHostToDevice);

	float time = 0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_add<<<fullBlocksPerGrid, blockSize >>>(dev_A, dev_B, dev_C);
	cudaEventRecord(stop);

	cudaMemcpy(C, dev_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::sub(float* A, float* B, float* C){
    init();
    cudaMemcpy(dev_A, A, 25 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B, B, 25 * sizeof(float), cudaMemcpyHostToDevice);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_sub<<< fullBlocksPerGrid, blockSize >>>(dev_A, dev_B, dev_C);
	cudaEventRecord(stop);

	cudaMemcpy( C , dev_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}

float Matrix_Math::mul(float* A, float* B, float* C){
    init();
	cudaMemcpy(dev_A, A, 25 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B, 25 * sizeof(float), cudaMemcpyHostToDevice);

    float time = 0;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_mul<<< fullBlocksPerGrid, blockSize >>>(dev_A, dev_B, dev_C);
	cudaEventRecord(stop);

	cudaMemcpy( C, dev_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	return time;
}
