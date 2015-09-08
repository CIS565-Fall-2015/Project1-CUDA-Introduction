#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include "matrix_math.h"

#define blockSize 64

/***********************************************
 * Kernel state (pointers are device pointers) *
 ***********************************************/
int N, mat_size;
float *dev_A, *dev_B, *dev_C;

//Helper functions
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

/**
*/
void cpyHostToDevice(const float *hst, float *dev) {
	cudaMemcpy(	/*destination*/ dev,
				/*source*/ hst,
				/*size in bytes to copy*/ mat_size,
				/*cudaMemcpy type*/ cudaMemcpyHostToDevice);

	checkCUDAErrorWithLine("Error copying memory from host to device");	
}

/**
*/
void cpyDeviceToHost(float *hst,const float *dev) {
	cudaMemcpy(	/*destination*/ hst,
				/*source*/ dev,
				/*size in bytes to copy*/ mat_size,
				/*cudaMemcpy type*/ cudaMemcpyDeviceToHost);

	checkCUDAErrorWithLine("Error copying memory from device to host");
}

/**
* Matrix addition
*/
__global__ void mat_add(float* A, float* B, float* C, int N) {
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N*N)
		C[index] = A[index] + B[index];
}

/**
* Matrix subtraction
*/
__global__ void mat_sub(float* A, float* B, float* C, int N) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < N*N)
		C[index] = A[index] - B[index];
}

/**
* Matrix multiplication
*/
__global__ void mat_mul(float* A, float* B, float* C, int N) {
	int index_A = 0;
	int index_B = 0;
	int index_C = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	int row = index_C/N;
	int col = index_C%N;

	if(index_C < N*N) {

		for(int i = 0; i < N; ++i) {
			index_A = i + row*N;
			index_B = col + i*N;
			C[index_C] += A[index_A] * B[index_B];
		}
		
	}
}

/**
* Allocate memory for device variables.
* Set the number of threads.
*/
void matrix_math::initMatrices(int dim) {
	//Set the dimensionality and size variables
	N = dim;
	mat_size = N*N*sizeof(float);

	//Allocate memory for the dev arrays
	//dev_A
	cudaMalloc((void**)&dev_A, mat_size);
	checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

	//dev_B
	cudaMalloc((void**)&dev_B, mat_size);
	checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

	//dev_C
	cudaMalloc((void**)&dev_C, mat_size);
	checkCUDAErrorWithLine("cudaMalloc dev_C failed!");
	
}

/**
*/
void matrix_math::mat_operation(float *hst_A, float *hst_B, float *hst_C, int op) {
	//copy matrices to device
	cpyHostToDevice(hst_A, dev_A);
	cpyHostToDevice(hst_B, dev_B);

	//The number of blocks
    dim3 blocksPerGrid((N + blockSize -1)/blockSize);
	
	//The number of threads per block
	dim3 threadsPerBlock(blockSize);
	cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
	switch (op) {
		case matrix_math::operation::ADD:
		mat_add<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);
		break;
		case matrix_math::operation::SUB:
		mat_sub<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);
		break;
		case matrix_math::operation::MUL:
		mat_mul<<<blocksPerGrid, threadsPerBlock>>>(dev_A, dev_B, dev_C, N);
		break;
		default:
		break;
	}

	cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
	float elapsedTime; 
    cudaEventElapsedTime(&elapsedTime , start, stop);
	printf("time is %f ms on the GPU", elapsedTime/100);

	//Copy the result to the host
	cpyDeviceToHost(hst_C, dev_C);
}