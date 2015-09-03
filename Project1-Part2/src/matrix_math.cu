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


float hos_A[] = {1,0,0,0,2
	,2,1,1,3,5
	,3,5,-1,0,1
	,7,6,5,4,3
	,0,0,7,2,2};

float hos_B[] = {1,1,2,5,3
	,0,0,1,1,0
	,1,0,1,0,4
	,-1,2,-5,3,-1
	,4,-2,-3,6,-9};

float hos_C[MATRIX_WIDTH*MATRIX_WIDTH];
float hos_D[MATRIX_WIDTH*MATRIX_WIDTH];

float * dev_A;
float * dev_B;
float * dev_C;






void init_input(float * A, float * B, int size)
{
	/*
	cudaMalloc((void**)&Ad, size);
	checkCUDAErrorWithLine("cudaMalloc Ad failed!");
	cudaMemcpy(Ad,A,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Bd, size);
	checkCUDAErrorWithLine("cudaMalloc Bd failed!");
	cudaMemcpy(Bd,B,size,cudaMemcpyHostToDevice);


	cudaMalloc((void**)&Cd, size);
	checkCUDAErrorWithLine("cudaMalloc Cd failed!");
	*/

	cudaMalloc((void**)&dev_A, size);
	checkCUDAErrorWithLine("cudaMalloc Ad failed!");
	cudaMemcpy(dev_A,A,size,cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_B, size);
	checkCUDAErrorWithLine("cudaMalloc Bd failed!");
	cudaMemcpy(dev_B,B,size,cudaMemcpyHostToDevice);


	cudaMalloc((void**)&dev_C, size);
	checkCUDAErrorWithLine("cudaMalloc Cd failed!");

	cudaThreadSynchronize();
}

void cpy_output(float * C, int size)
{
	//cudaMemcpy(C,Cd,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(C,dev_C,size,cudaMemcpyDeviceToHost);

	//free
	//cudaFree(Cd);
	//cudaFree(Ad);
	//cudaFree(Bd);
	cudaFree(dev_C);
	cudaFree(dev_A);
	cudaFree(dev_B);
}


__global__ void MatrixAddKernel(float * Ad, float * Bd, float * Cd, int width)
{
	int index = threadIdx.x;

	Cd[index] = Ad[index] + Bd[index];

	//printf("%f\n",Cd[index]);
}

void mat_add (float * A, float * B, float * C, int width = 5)
{
	//square matrix only
	//C = A + B
	int size = width * width * sizeof(float);

	init_input(A,B,size);
	
	//kernel
	dim3 dimBlock(width*width);
	dim3 dimGrid(1,1);
	//MatrixAddKernel<<< dimGrid , dimBlock >>>(Ad,Bd,Cd,width);
	MatrixAddKernel<<< dimGrid , dimBlock >>>(dev_A,dev_B,dev_C,width);

	//transfer from device to host and free
	cpy_output(C,size);
}


__global__ void MatrixSubKernel(float * Ad, float * Bd, float * Cd, int width)
{
	int index = threadIdx.x;

	Cd[index] = Ad[index] - Bd[index];
}

void mat_sub (float * A, float * B, float * C, int width = 5)
{
	//square matrix only
	//C = A - B
	int size = width * width * sizeof(float);

	init_input(A,B,size);

	//kernel
	dim3 dimBlock(width*width);
	dim3 dimGrid(1,1);
	MatrixSubKernel<<< dimGrid , dimBlock >>>(dev_A,dev_B,dev_C,width);

	//transfer from device to host and free
	cpy_output(C,size);
}


__global__ void MatrixMulKernel(float * Ad, float * Bd, float * Cd, int width)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	float Cvalue = 0.0f;

	for(int k = 0; k < width; k++)
	{
		Cvalue += (Ad[ty * width + k]) * (Bd[k * width + tx]);
	}

	Cd[ty * width + tx] = Cvalue;
}



void mat_mul (float * A, float * B, float * C, int width = 5)
{
	//square matrix only
	//C = A * B
	int size = width * width * sizeof(float);

	init_input(A,B,size);

	//kernel
	dim3 dimBlock(width,width);
	dim3 dimGrid(1,1);
	MatrixMulKernel<<< dimGrid , dimBlock >>>(dev_A,dev_B,dev_C,width);


	//transfer from device to host and free
	cpy_output(C,size);
}


void init()
{
	printMatrix(hos_A,MATRIX_WIDTH);
	printf("\n");
	printMatrix(hos_B,MATRIX_WIDTH);
	printf("\n");
}

void free()
{
	printf("add:\n");
	printMatrix(hos_C,MATRIX_WIDTH);
	printf("\n");

	printf("mul:\n");
	printMatrix(hos_D,MATRIX_WIDTH);
	printf("\n");
}

void run()
{
	mat_add(hos_A,hos_B,hos_C,MATRIX_WIDTH);
	mat_mul(hos_A,hos_B,hos_D,MATRIX_WIDTH);
}

void printMatrix (float * M, int width)
{
	for(int i = 0; i < width; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if(j !=0 )
			{
				printf(",");
			}
			printf("%f",M[i*width+j]);
		}
		printf("\n");
	}
}