#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
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

#define blockSize 128
dim3 threadsPerBlock(blockSize);
float *dev_mat_a;
float *dev_mat_b;
float *dev_mat_c;

/**
 * Initialize memory, update some globals
 */
void Matrix_Math::initialize(int N) {
	
    int total = N * N;
    dim3 fullBlocksPerGrid((total + blockSize - 1) / blockSize);
	
    cudaMalloc((void**)&dev_mat_a, total * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_mat_a failed!");

    cudaMalloc((void**)&dev_mat_b, total * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_mat_b failed!");

    cudaMalloc((void**)&dev_mat_c, total * sizeof(float));
    checkCUDAErrorWithLine("cudaMalloc dev_mat_c failed!");

}

void Matrix_Math::cleanUp() {
    cudaFree(dev_mat_a);
    cudaFree(dev_mat_b);
    cudaFree(dev_mat_c);
}

__global__ void mat_add(float *A, float *B, float *C, int N) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N*N) {
		C[index] = A[index] + B[index];
	}
}

__global__ void mat_sub(float *A, float *B, float *C, int N) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N*N) {
		C[index] = A[index] - B[index];
	}
}

__global__ void mat_mul(float *A, float *B, float *C, int N) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N*N) {
		int row = index / N;
		int column = index % N;
		C[index] = 0;

		for(int i = 0; i < N; i++) {
			C[index] += A[i + row*N] * B[column + i*N];
		}
	}
}

void Matrix_Math::kernMatAdd(int N, float *hst_mat_a, float *hst_mat_b, float *hst_mat_c) {
	int total = N * N;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(dev_mat_a, hst_mat_a, total * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, hst_mat_b, total * sizeof(float), cudaMemcpyHostToDevice);
    dim3 fullBlocksPerGrid((total + blockSize - 1) / blockSize);

	cudaEventRecord(start);
	mat_add<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_mat_a, dev_mat_b, dev_mat_c, N);
	cudaEventRecord(stop);

	cudaMemcpy(hst_mat_c, dev_mat_c, total * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << milliseconds << "ms for adding" << std::endl;
}

void Matrix_Math::kernMatSub(int N, float *hst_mat_a, float *hst_mat_b, float *hst_mat_c) {
    int total = N * N;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(dev_mat_a, hst_mat_a, total * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, hst_mat_b, total * sizeof(float), cudaMemcpyHostToDevice);
    dim3 fullBlocksPerGrid((total + blockSize - 1) / blockSize);

	cudaEventRecord(start);
	mat_sub<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_mat_a, dev_mat_b, dev_mat_c, N);
	cudaEventRecord(stop);

	cudaMemcpy(hst_mat_c, dev_mat_c, total * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << milliseconds << "ms for subtracting" << std::endl;
}

void Matrix_Math::kernMatMul(int N, float *hst_mat_a, float *hst_mat_b, float *hst_mat_c) {
    int total = N * N;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(dev_mat_a, hst_mat_a, total * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_b, hst_mat_b, total * sizeof(float), cudaMemcpyHostToDevice);
    dim3 fullBlocksPerGrid((total + blockSize - 1) / blockSize);

	cudaEventRecord(start);
	mat_mul<<<fullBlocksPerGrid, threadsPerBlock>>>(dev_mat_a, dev_mat_b, dev_mat_c, N);
	cudaEventRecord(stop);

	cudaMemcpy(hst_mat_c, dev_mat_c, total * sizeof(float), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	std::cout << milliseconds << "ms for multiplying" << std::endl;
}