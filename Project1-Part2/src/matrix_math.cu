#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "matrix_math.h"
#include "utilityCore.hpp"

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

#define N 5

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

float *dev_mat1;
float *dev_mat2;
float *hos_mat1;
float *hos_mat2;

/******************
* Setup & tear down *
******************/

/**
* Initialize memory, update some globals
*/
void MMath::init() {

	cudaMalloc((void**)&dev_mat1, N * N * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat1 failed!");

	cudaMalloc((void**)&dev_mat2, N * N * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat2 failed!");

	hos_mat1 = (float *)malloc(N * N * sizeof(float));
	hos_mat2 = (float *)malloc(N * N * sizeof(float));

	cudaMemcpy(hos_mat1, dev_mat1, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy hos_mat1 failed!");

	cudaMemcpy(hos_mat2, dev_mat2, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy hos_mat2 failed!");

	//test();
	test2();
	test3();

	/*
	1 Block, 5x5 Thread: A:0.006976, S:0.004736, M:0.007200
	5 Block, 5x1 Thread: A:0.004704, S:0.004672, M:0.007264
	*/
}

void MMath::terminate() {
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);
	free(hos_mat1);
	free(hos_mat2);
}


/******************
* Kernels *
******************/

/**
* Addition
*/

__global__ void mat_add(float *A, float *B, float *C){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	C[j*N + i] = A[j*N + i] + B[j*N + i];
}

/**
* Subtraction
*/
__global__ void mat_sub(float *A, float *B, float *C){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	C[j*N + i] = A[j*N + i] - B[j*N + i];
}

/**
* Multiplication
*/
__global__ void mat_mul(float *A, float *B, float *C){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;

	float p = 0;

	for (int k = 0; k < N; k++){
		p += A[j*N + k] * B[k*N + i];
	}

	C[j*N + i] = p;
}

void MMath::test(){
	for (int i = 0; i < N*N; i++){
		hos_mat1[i] = 1.0f;
		hos_mat2[i] = 2.0f;
	}

	float *dev_matp;
	float *hos_matp;

	cudaMalloc((void**)&dev_matp, N * N * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_matp failed!");

	hos_matp = (float *)malloc(N * N * sizeof(float));

	// Test add
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 gridSize(1, 1);
	dim3 blockSize(N, N);

	printf("Addition result:\n");
	mat_add<<<gridSize, blockSize>>>(dev_mat1, dev_mat2, dev_matp);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy hos_matp failed!");

	for (int i = 0; i < N*N; i++) {
		printf("%f ", hos_matp[i]);
		printf("\n");
	}

	// Test sub
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	printf("Subtraction result:\n");
	mat_sub<<<gridSize, blockSize>>>(dev_mat1, dev_mat2, dev_matp);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy hos_matp failed!");

	for (int i = 0; i < N*N; i++) {
		printf("%f ", hos_matp[i]);
		printf("\n");
	}

	// Test mul
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	printf("Multiplication result:\n");
	mat_mul<<<gridSize, blockSize>>>(dev_mat1, dev_mat2, dev_matp);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy hos_matp failed!");

	for (int i = 0; i < N*N; i++) {
		printf("%f ", hos_matp[i]);
		printf("\n");
	}

	cudaFree(dev_matp);
	free(hos_matp);
}

void MMath::test2(){
	for (int i = 0; i < N*N; i++){
		hos_mat1[i] = 1.0f;
		hos_mat2[i] = 2.0f;
	}

	float *dev_matp;
	float *hos_matp;

	cudaMalloc((void**)&dev_matp, N * N * sizeof(float));
	hos_matp = (float *)malloc(N * N * sizeof(float));

	cudaEvent_t start, stop;

	// --------------------------------------------------------------------------------

	dim3 gridSize(1, 1);
	dim3 blockSize(N, N);

	// Test add
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_add << <gridSize, blockSize >> >(dev_mat1, dev_mat2, dev_matp);
	cudaEventRecord(stop);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float msAdd1 = 0;
	cudaEventElapsedTime(&msAdd1, start, stop);

	// Test sub
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_sub << <gridSize, blockSize >> >(dev_mat1, dev_mat2, dev_matp);
	cudaEventRecord(stop);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float msSub1 = 0;
	cudaEventElapsedTime(&msSub1, start, stop);

	// Test mul
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_mul << <gridSize, blockSize >> >(dev_mat1, dev_mat2, dev_matp);
	cudaEventRecord(stop);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float msMul1 = 0;
	cudaEventElapsedTime(&msMul1, start, stop);

	cudaFree(dev_matp);
	free(hos_matp);

	printf("1 Block, 5x5 Thread: A:%f, S:%f, M:%f\n", msAdd1, msSub1, msMul1);
}

void MMath::test3(){
	for (int i = 0; i < N*N; i++){
		hos_mat1[i] = 1.0f;
		hos_mat2[i] = 2.0f;
	}

	float *dev_matp;
	float *hos_matp;

	cudaMalloc((void**)&dev_matp, N * N * sizeof(float));
	hos_matp = (float *)malloc(N * N * sizeof(float));

	cudaEvent_t start, stop;

	// --------------------------------------------------------------------------------

	dim3 gridSize(1, 5);
	dim3 blockSize(N, 1);

	// Test add
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_add << <gridSize, blockSize >> >(dev_mat1, dev_mat2, dev_matp);
	cudaEventRecord(stop);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float msAdd1 = 0;
	cudaEventElapsedTime(&msAdd1, start, stop);

	// Test sub
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_sub << <gridSize, blockSize >> >(dev_mat1, dev_mat2, dev_matp);
	cudaEventRecord(stop);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float msSub1 = 0;
	cudaEventElapsedTime(&msSub1, start, stop);

	// Test mul
	cudaMemcpy(dev_mat1, hos_mat1, N * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hos_mat2, N * N * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_mul << <gridSize, blockSize >> >(dev_mat1, dev_mat2, dev_matp);
	cudaEventRecord(stop);

	cudaMemcpy(hos_matp, dev_matp, N * N * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float msMul1 = 0;
	cudaEventElapsedTime(&msMul1, start, stop);

	cudaFree(dev_matp);
	free(hos_matp);

	printf("5 Block, 5x1 Thread: A:%f, S:%f, M:%f\n", msAdd1, msSub1, msMul1);
}