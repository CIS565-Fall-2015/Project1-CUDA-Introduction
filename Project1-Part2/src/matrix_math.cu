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
float *hst_mat_c;

/***********************************************
* Device state *
***********************************************/

int width;
dim3 threadsPerBlock(blockSize);

float *dev_mat_a;
float *dev_mat_b;
float *dev_mat_c;


/******************
* initialization *
******************/

void MatrixMath::initialization(int mat_width) {
	width = mat_width;
	dim3 block_dim(mat_width, mat_width);
	dim3 grid_dim(1, 1);

	// Is it this memory I am going to have to move?
	// use malloc then move this to the device
	//what am i initiallizing hte values too here?
	cudaMalloc((void**)&hst_mat_a, (mat_width * mat_width) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc hst_mat_a failed!");

	cudaMalloc((void**)&hst_mat_b, (mat_width * mat_width) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc hst_mat_b failed!");

	//think I only want to allocate on the devuce when doing cuda

	cudaMalloc((void**)&dev_mat_a, (mat_width * mat_width) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat_a failed!");

	cudaMalloc((void**)&dev_mat_b, (mat_width * mat_width) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat_b failed!");
}

void MatrixMath::cleanup() {
	// free memory here, call when done
}

__global__ void kern_mat_add(float *A, float *B, float *C) {
	//so this is just gonna be dealing with one calculation, not hte whole matrix
}

__global__ void kern_mat_sub(float *A, float *B, float *C) {

}

__global__ void kern_mat_mul(float *A, float *B, float *C) {

}

void MatrixMath::mat_add(float *A, float *B, float *C) {
	// first copy to device memory, then envoke kernel function
	cudaMemcpy(dev_mat_a, A, (width * width) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, B, (width * width) * sizeof(float), cudaMemcpyHostToDevice);
	kern_mat_add<<<grid_dim, block_dim>>>(dev_mat_a, dev_mat_b, dev_mat_c);
	cudaMemcpy(dev_mat_c, C, (width * width) * sizeof(float), cudaMemcpyDeviceToHost);
}

void MatrixMath::mat_sub(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_a, A, (width * width) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, B, (width * width) * sizeof(float), cudaMemcpyHostToDevice);
	kern_mat_sub<<<grid_dim, block_dim>>>(dev_mat_a, dev_mat_b, dev_mat_c);
	cudaMemcpy(dev_mat_c, C, (width * width) * sizeof(float), cudaMemcpyDeviceToHost);
}

void MatrixMath::mat_mul(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_a, A, (width * width) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, B, (width * width) * sizeof(float), cudaMemcpyHostToDevice);
	kern_mat_mul<<<grid_dim, block_dim>>>(dev_mat_a, dev_mat_b, dev_mat_c);
	cudaMemcpy(dev_mat_c, C, (width * width) * sizeof(float), cudaMemcpyDeviceToHost);
}

/*
	This is where we will run tests to confirm the functions work
*/
void MatrixMath::run_tests() {
	cleanup();
}