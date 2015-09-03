#define GLM_FORCE_CUDA
#include <iostream>
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
int hst_width;

float *hst_mat_a;
float *hst_mat_b;
float *hst_mat_c;

/***********************************************
* Device state *
***********************************************/


dim3 block_dim;
dim3 grid_dim;

float *dev_mat_a;
float *dev_mat_b;
float *dev_mat_c;


/******************
* initialization *
******************/

void MatrixMath::initialization(int mat_width) {
	hst_width = mat_width;
	// TODO: Not sure on how to set this up, cause we are using a 1D array, not 2D. Think this is valid
	block_dim = dim3(mat_width * mat_width);
	grid_dim = dim3(1, 1);

	// Is it this memory I am going to have to move?
	// use malloc then move this to the device
	//what am i initiallizing hte values too here?
	hst_mat_a = (float*)malloc((mat_width * mat_width) * sizeof(float));
	// TODO: Error check

	hst_mat_b = (float*)malloc((mat_width * mat_width) * sizeof(float));
	// TODO: Error check

	hst_mat_c = (float*)malloc((mat_width * mat_width) * sizeof(float));
	// TODO: Error check

	//think I only want to allocate on the device when doing cuda

	cudaMalloc((void**)&dev_mat_a, (mat_width * mat_width) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat_a failed!");

	cudaMalloc((void**)&dev_mat_b, (mat_width * mat_width) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat_b failed!");

	cudaMalloc((void**)&dev_mat_c, (mat_width * mat_width) * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_mat_c failed!");
}

void MatrixMath::cleanup() {
	// Why is freeing the host memory throwing errors?
	//free(hst_mat_a);
	//free(hst_mat_b);
	//free(hst_mat_c);

	cudaFree(dev_mat_a);
	cudaFree(dev_mat_b);
	cudaFree(dev_mat_c);
}

__global__ void kern_mat_add(float *A, float *B, float *C, int width) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

__global__ void kern_mat_sub(float *A, float *B, float *C, int width) {
	int i = threadIdx.x;
	C[i] = A[i] - B[i];
}

__global__ void kern_mat_mul(float *A, float *B, float *C, int width) {
	//int i = threadIdx.x;
	int i = threadIdx.x % width;
	int j = threadIdx.x / width;
	float Ci = 0.0f;
	//this is totally wrong?
	for (int k = 0; k < width; k++) {
		float Ai = A[j * width + k];
		float Bi = B[k * width + i];

		Ci += Ai * Bi;
	}
	C[j * width + i] = Ci;
}

void MatrixMath::mat_add(float *A, float *B, float *C) {
	// first copy to device memory, then envoke kernel function
	cudaMemcpy(dev_mat_a, A, (hst_width * hst_width) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, B, (hst_width * hst_width) * sizeof(float), cudaMemcpyHostToDevice);
	kern_mat_add<<<grid_dim, block_dim>>>(dev_mat_a, dev_mat_b, dev_mat_c, hst_width);
	cudaMemcpy(C, dev_mat_c, (hst_width * hst_width) * sizeof(float), cudaMemcpyDeviceToHost);
}

void MatrixMath::mat_sub(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_a, A, (hst_width * hst_width) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, B, (hst_width * hst_width) * sizeof(float), cudaMemcpyHostToDevice);
	kern_mat_sub<<<grid_dim, block_dim>>>(dev_mat_a, dev_mat_b, dev_mat_c, hst_width);
	cudaMemcpy(C, dev_mat_c, (hst_width * hst_width) * sizeof(float), cudaMemcpyDeviceToHost);
}

void MatrixMath::mat_mul(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_a, A, (hst_width * hst_width) * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, B, (hst_width * hst_width) * sizeof(float), cudaMemcpyHostToDevice);
	kern_mat_mul<<<grid_dim, block_dim>>>(dev_mat_a, dev_mat_b, dev_mat_c, hst_width);
	cudaMemcpy(C, dev_mat_c, (hst_width * hst_width) * sizeof(float), cudaMemcpyDeviceToHost);
}

void MatrixMath::print_mat(float *mat, int width) {
	for (int i = 0; i < width; i++) {
		for (int j = 0; j < width; j++) {
			fprintf(stdout, "%f, ", mat[(i * width) + j]);
		}
		fprintf(stdout, "\n");
	}
}

/*
	This is where we will run tests to confirm the functions work
*/
void MatrixMath::run_tests() {
	float A[] = {
		9.0f, 10.0f, 2.0f, 1.0f, 7.5f,
		2.0f, 1.0f, 1.0f, 1.0f, 1.0f,
		1.1f, 3.0f, 1.0f, 20.0f, 13.0f,
		6.6f, 3.0f, 0.0f, 1.0f, 9.0f,
		2.0f, 4.0f, 8.0f, 4.5f, 0.0f,
	};
	hst_mat_a = A;

	float B[] = {
		2.0f, 1.0f, 3.0f, 5.0f, 9.0f,
		0.5f, 0.1f, 19.0f, 2.0f, 12.0f,
		5.0f, 2.3f, 8.0f, 2.0f, 13.0f,
		6.0f, 4.5f, 9.0f, 1.0f, 0.75f,
		11.0f, 11.0f, 7.8f, 22.0f, 1.0f,
	};
	hst_mat_b = B;

	float C[] = {
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
	};
	hst_mat_c = C;

	fprintf(stdout, "Running matrix addition, subtraction, and multiplcation tests...\n\n");

	fprintf(stdout, "Matrix A:\n");
	MatrixMath::print_mat(A, hst_width);

	fprintf(stdout, "\n\n");

	fprintf(stdout, "Matrix B:\n");
	MatrixMath::print_mat(B, hst_width);

	fprintf(stdout, "\n\n");

	fprintf(stdout, "Addition Test A + B = \n");
	MatrixMath::mat_add(A, B, C);
	MatrixMath::print_mat(C, hst_width);

	fprintf(stdout, "\n\n");

	fprintf(stdout, "Subtraction Test A - B = \n");
	MatrixMath::mat_sub(A, B, C);
	MatrixMath::print_mat(C, hst_width);

	fprintf(stdout, "\n\n");

	fprintf(stdout, "Multiplication Test A * B = \n");
	MatrixMath::mat_mul(A, B, C);
	MatrixMath::print_mat(C, hst_width);

	cleanup();
}