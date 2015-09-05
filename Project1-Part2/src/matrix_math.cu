#include "matrix_math.h"
#include<cuda.h>

float *dev_A,*dev_B,*dev_C;

void Matrix_Math::initiate(int size,float *A,float *B){
	cudaMalloc((void**)&dev_A, size*size * sizeof(float));
	cudaMalloc((void**)&dev_B, size*size * sizeof(float));
	cudaMalloc((void**)&dev_C, size*size * sizeof(float));
	cudaMemcpy(dev_A,A,size*size*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B,B,size*size*sizeof(float),cudaMemcpyHostToDevice);
}

void Matrix_Math::copyBack(int size,float *C){
	cudaMemcpy(C,dev_C,size*size*sizeof(float),cudaMemcpyDeviceToHost);
}

__global__ void mat_add(int N,float *A,float *B,float *C){
	int index=blockIdx.x*blockDim.x*blockDim.y+blockDim.x*threadIdx.y+threadIdx.x;
	if(index<N*N)
		C[index]=A[index]+B[index];
}

__global__ void mat_sub(int N,float *A,float *B,float *C){
	int index=blockIdx.x*blockDim.x*blockDim.y+blockDim.x*threadIdx.y+threadIdx.x;
	if(index<N*N)
		C[index]=A[index]-B[index];
}

__global__ void mat_mul(int n,float *A,float *B,float *C){
	int index=blockIdx.x*blockDim.x*blockDim.y+blockDim.x*threadIdx.y+threadIdx.x;
	int result=0;
	for(int i=0;i<n;++i){
		result+=A[blockDim.x*threadIdx.y+i]*B[blockDim.x*i+threadIdx.x];
	}
	C[index]=result;
}

float Matrix_Math::add(int inputSize,int blockSize,float *A,float *B,float *C){
	cudaEvent_t start, stop;
	initiate(inputSize,A,B);
	dim3 blockNum=(1,1+inputSize*inputSize/blockSize/blockSize);
	dim3 block(blockSize,blockSize);
	cudaEventCreate(&start);
	mat_add<<<blockNum,block>>>(inputSize,dev_A,dev_B,dev_C);
	cudaEventCreate(&stop);
	copyBack(inputSize,C);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	return cudaEventElapsedTime(&milliseconds, start, stop);
}

float Matrix_Math::sub(int inputSize,int blockSize,float *A,float *B,float *C){
	cudaEvent_t start, stop;
	initiate(inputSize,A,B);
	dim3 blockNum=(1,1+inputSize*inputSize/blockSize/blockSize);
	dim3 block(blockSize,blockSize);
	cudaEventCreate(&start);
	mat_sub<<<blockNum,block>>>(inputSize,dev_A,dev_B,dev_C);
	cudaEventCreate(&stop);
	copyBack(inputSize,C);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	return cudaEventElapsedTime(&milliseconds, start, stop);
}

float Matrix_Math::mul(int inputSize,int blockSize,float *A,float *B,float *C){
	cudaEvent_t start, stop;
	initiate(inputSize,A,B);
	dim3 blockNum=(1,1+inputSize*inputSize/blockSize/blockSize);
	dim3 block(blockSize,blockSize);
	cudaEventCreate(&start);
	mat_mul<<<blockNum,block>>>(inputSize,dev_A,dev_B,dev_C);
	cudaEventCreate(&stop);
	copyBack(inputSize,C);
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	return cudaEventElapsedTime(&milliseconds, start, stop);
}