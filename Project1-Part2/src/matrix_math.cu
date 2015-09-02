#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
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


/***********************************************
* Host state *
***********************************************/

float *hst_mat_a;
float *hst_mat_b;

/***********************************************
* Device state *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

float *dev_mat_a;
float *dev_mat_b;


/******************
* initialization *
******************/

void MatrixMath::initialization(int mat_width, int mat_height) {
	// Is it this memory I am going to have to move?
	// use malloc then move this to the device
	//what am i initiallizing hte values too here?
	cudaMalloc((void**)&hst_mat_a, (mat_width * mat_height) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc hst_mat_a failed!");

	cudaMalloc((void**)&hst_mat_b, (mat_width * mat_height) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc hst_mat_b failed!");

	//think I only want to allocate on the devuce when doing cuda

	cudaMalloc((void**)&dev_mat_a, (mat_width * mat_height) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat_a failed!");

	cudaMalloc((void**)&dev_mat_b, (mat_width * mat_height) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat_b failed!");
}

// TODO: Create a function to free the memory we allocated

__global__ void kern_mat_add(float *A, float *B, float *C) {

}

__global__ void kern_mat_sub(float *A, float *B, float *C) {

}

__global__ void kern_mat_mul(float *A, float *B, float *C) {

}

void MatrixMath::mat_add(float *A, float *B, float *C) {

}

void MatrixMath::mat_sub(float *A, float *B, float *C) {

}

void MatrixMath::mat_mul(float *A, float *B, float *C) {

}