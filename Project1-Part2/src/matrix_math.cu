#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>

#include "matrix_math.h"

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define mat_size 5
#define block_size 1
/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/


//dim3 threadsPerBlock(blockSize);

float *dev_MA;
float *dev_MB;
float *dev_MC;

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
/************************************************************************
*initialize two 5x5 matrices on the host and two on the device *********
************************************************************************/
void NMatrix::initialization(float *hst_MA, float *hst_MB) {

	float size_z = mat_size * mat_size * sizeof(float);

	cudaMalloc((void**)&dev_MA, size_z);
	checkCUDAErrorWithLine("cudaMalloc dev_MA failed!");

	cudaMalloc((void**)&dev_MB, size_z);
	checkCUDAErrorWithLine("cudaMalloc dev_MB failed!");
	cudaMalloc((void**)&dev_MC, size_z);
	checkCUDAErrorWithLine("cudaMalloc dev_MC failed!");

	cudaMemcpy(dev_MA, hst_MA, size_z, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy hst_MA failed!");

	cudaMemcpy(dev_MB, hst_MB, size_z, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy hst_MA failed!");
}
/*implement function*/
__global__ void Add(float* A, float *B, float *C){
//__global__ void Add(float A[mat_size][mat_size], float B[mat_size][mat_size], float C[mat_size][mat_size]){
	int i = threadIdx.x;
	int j = threadIdx.y;
	int index = i*mat_size + j;
	if (i < mat_size && j < mat_size)
		//C[i][j] = A[i][j] + B[i][j];
		C[index] = A[index] + B[index];
	
}

__global__ void Sub(float *A, float* B, float *C){
	int id = threadIdx.x*mat_size+threadIdx.y;
	C[id] = A[id] - B[id];
}

__global__ void Mul(float *A, float *B, float *C){
	int id = threadIdx.x*mat_size + threadIdx.y;

	for (int k = 0; k < mat_size; k++){
		//C[i][j]+ = A[i][k] * B[k][j];
	    //i=threadIdx.x,j=threadIdx.y
		C[id] += A[threadIdx.x*mat_size + k] * B[k*mat_size + threadIdx.y];
    }
	
}
void NMatrix::mat_add(float *hst_A, float * hst_B, float* hst_C){
	

	dim3 threadsPerBlock(mat_size, mat_size);
	Add <<< block_size, threadsPerBlock >> >(dev_MA, dev_MB, dev_MC);
	cudaMemcpy(hst_C, dev_MC,  mat_size*mat_size*sizeof(float), cudaMemcpyDeviceToHost);//destination,source,
	endMAtrix();
}
void NMatrix::mat_sub(float  *hst_A, float *hst_B, float* hst_C){
	
	dim3 threadsPerBlock(mat_size, mat_size);
	Sub <<< block_size, threadsPerBlock >>>(dev_MA, dev_MB, dev_MC);
	cudaMemcpy(hst_C, dev_MC, mat_size*mat_size*sizeof(float), cudaMemcpyDeviceToHost);//destination,source,
	endMAtrix();
}

void NMatrix::mat_mul(float  *hst_A, float * hst_B, float* hst_C){
//	C = A * B;
	
	dim3 threadsPerBlock(mat_size, mat_size);
	Mul <<< block_size, threadsPerBlock >>>(dev_MA, dev_MB, dev_MC);
	cudaMemcpy(hst_C, dev_MC, mat_size*mat_size*sizeof(float), cudaMemcpyDeviceToHost);//destination,source,
	endMAtrix();
}

void NMatrix::endMAtrix()
{
	cudaFree(dev_MA);
	cudaFree(dev_MB);
	cudaFree(dev_MC);
}