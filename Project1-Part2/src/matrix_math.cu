#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
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


int arraySize;
int width;

float *hst_A;
float *hst_B;
float *hst_C;
float *dev_A;
float *dev_B;
float *dev_C;

void generateRandomMatrix(int arraySize, float * mat) {
	for (int i = 0; i < arraySize; i++) {

		mat[i] = rand() % 10 + 1;
	}
}

/**
 * Initialize memory, update some globals
 */
void MatMath::initSimulation(int N) {
	width = N;
    arraySize = N*N;

    hst_A = (float*) malloc (arraySize * sizeof(float));

    hst_B = (float*) malloc (arraySize * sizeof(float));

	hst_C = (float*) malloc (arraySize * sizeof(float));

    cudaMalloc((void**)&dev_A, arraySize * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

	cudaMalloc((void**)&dev_B, arraySize * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

	cudaMalloc((void**)&dev_C, arraySize * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_C failed!");

    generateRandomMatrix(arraySize, hst_A);

	generateRandomMatrix(arraySize, hst_B);

    cudaThreadSynchronize();
}

/**
 * Creating kernel function to add matrices
 */
__global__ void kern_mat_add(float * A, float *B, float * C) {
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}

__global__ void kern_mat_sub(float * A, float *B, float * C) {
	int i = threadIdx.x;
	C[i] = A[i] - B[i];
}

__global__ void kern_mat_mul(float * A, float *B, float * C, int width) {
	int x = threadIdx.x;
	int y = threadIdx.y;

	float p = 0.0f;

	for (int k = 0; k < width; k++) {
		float m = A[y * width + k];
		float n = B[k*width + x];
		p += m * n;
	}

	C[y * width + x] = p;
}

void MatMath::mat_add(float * A, float *B, float * C) {
	int size = arraySize * sizeof(float);
	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, hst_B, size, cudaMemcpyHostToDevice);

	kern_mat_add<<<1, arraySize>>>(dev_A, dev_B, dev_C);

	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);
}

void MatMath::mat_sub(float * A, float *B, float * C) {
	int size = arraySize * sizeof(float);
	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, hst_B, size, cudaMemcpyHostToDevice);

	kern_mat_sub<<<1, arraySize>>>(dev_A, dev_B, dev_C);

	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);
}

void MatMath::mat_mul(float * A, float *B, float * C) {
	int size = arraySize * sizeof(float);
	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, hst_B, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);
	kern_mat_mul<<<dimGrid, dimBlock>>>(dev_A, dev_B, dev_C, width);

	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);
}

void MatMath::testFunc(int test) {
	int i;
	for (i=0;i < arraySize;i++) {
		if (i%width == 0) {
			printf("\n");
		}
		printf("%f ",hst_A[i]);
		
	}
	printf("\n");
	for (i=0;i < arraySize;i++) {
		if (i%width == 0) {
			printf("\n");
		}
		printf("%f ",hst_B[i]);
		
	}
	if (test == 0) {
		MatMath::mat_add(hst_A, hst_B, hst_C);
	}
	else if (test == 1) {
		MatMath::mat_sub(hst_A, hst_B, hst_C);
	}
	else if (test == 2) {
		MatMath::mat_mul(hst_A, hst_B, hst_C);
	}
	printf("\n");
	for (i=0;i < arraySize;i++) {
		if (i%width == 0) {
			printf("\n");
		}
		printf("%f ",hst_C[i]);
		
	}
}