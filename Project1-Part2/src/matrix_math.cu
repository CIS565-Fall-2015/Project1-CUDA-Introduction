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


float *dev_matA;
float *dev_matB;
float *dev_matC;

void MatrixCalc::initMats(float *hst_matA,float *hst_matB,int matWidth)
{
	int size = matWidth*matWidth*sizeof(float);

	cudaMalloc((void**)&dev_matA, size);
	checkCUDAErrorWithLine("cudaMalloc dev_matA failed!");

	cudaMalloc((void**)&dev_matB, size);
	checkCUDAErrorWithLine("cudaMalloc dev_matB failed!");

	cudaMemcpy(dev_matA,hst_matA,size,cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy hst_matA to dev_matA failed!");

	cudaMemcpy(dev_matB, hst_matB, size, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy hst_matB to dev_matB failed!");

	cudaMalloc((void**)&dev_matC, size);
	checkCUDAErrorWithLine("cudaMalloc dev_matB failed!");
	//TODO later: try seperate malloc and memcpy

}


__global__ void kernMatAdd(float *matA,float *matB,float *matC,int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idx =  tx*width + ty;
	matC[idx] = matA[idx] + matB[idx];
	
}

void MatrixCalc::mat_add(float*A, float*B, float*C,int width)
{
	initMats(A,B,width);//todo later: 5
	dim3 threadsPerBlock(width, width);
	kernMatAdd<<<1,threadsPerBlock>>>(dev_matA,dev_matB,dev_matC,width);
	cudaMemcpy(C,dev_matC,width*width*sizeof(float),cudaMemcpyDeviceToHost);
	freeMats();
}


__global__ void kernMatSub(float *matA, float *matB, float *matC, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idx = tx*width + ty;
	matC[idx] = matA[idx] - matB[idx];

}

void MatrixCalc::mat_sub(float*A, float*B, float*C, int width)
{
	initMats(A, B, width);//todo later: 5
	dim3 threadsPerBlock(width, width);
	kernMatSub <<<1, threadsPerBlock >>>(dev_matA, dev_matB, dev_matC, width);
	cudaMemcpy(C, dev_matC, width*width*sizeof(float), cudaMemcpyDeviceToHost);
	freeMats();
}

__global__ void kernMatMul(float *matA, float *matB, float *matC, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int idx = tx*width + ty;

	float sum = 0;
	for (int i = 0; i < width; i++)
	{
		sum += matA[tx*width + i] * matB[i*width + ty];
	}
	matC[idx] = sum;

}

void MatrixCalc::mat_mul(float*A, float*B, float*C, int width)
{
	initMats(A, B, width);//todo later: 5
	dim3 threadsPerBlock(width, width);
	kernMatMul <<<1, threadsPerBlock >>>(dev_matA, dev_matB, dev_matC, width);
	cudaMemcpy(C, dev_matC, width*width*sizeof(float), cudaMemcpyDeviceToHost);
	freeMats();
}

void MatrixCalc::freeMats()
{
	cudaFree(dev_matA);
	cudaFree(dev_matB);
	cudaFree(dev_matC);
}