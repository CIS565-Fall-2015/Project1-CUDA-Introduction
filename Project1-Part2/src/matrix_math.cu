#include "matrix_math.hpp"

float* hst_A;
float* hst_B;
float* dev_A;
float* dev_B;

// DEVICE FUNCTIONS
__global__ void mat_add(float* A, float* B, float* C, int width){
	int i = threadIdx.x;
	int j = threadIdx.y;

	int ind = i + width*j;
	
	if (ind < width*width){
		C[ind] = A[ind] + B[ind];
	}
}

__global__ void mat_sub(float* A, float* B, float* C, int width){
	int i = threadIdx.x;
	int j = threadIdx.y;

	int ind = i + width*j;
	
	if (ind < width*width){
		C[ind] = A[ind] - B[ind];
	}
}

__global__ void mat_mul(float* A, float* B, float* C, int width){
	int i = threadIdx.x;
	int j = threadIdx.y;

	float val = 0;
	
	for (int k = 0; k < width; ++k){
		float Ael = A[k + j*width];
		float Bel = B[i + k*width];
		val += Ael * Bel;
	}

	C[i + j*width] = val;
}

// KERNEL FUNCTIONS
void kern_mat_add(float* A, float* B, float* C, int width){
	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	mat_add<<<dimGrid, dimBlock>>>(A, B, C, width);
}

void kern_mat_sub(float* A, float* B, float* C, int width){
	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	mat_sub<<<dimGrid, dimBlock>>>(A, B, C, width);
}

void kern_mat_mul(float* A, float* B, float* C, int width){
	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	mat_mul<<<dimGrid, dimBlock>>>(A, B, C, width);
}

void initialize(int width){
	//int width = 2;
	int size = width * width * sizeof(float);

	hst_A = (float*)malloc(size);
	hst_B = (float*)malloc(size);

	for (int i=0; i < width*width; i++){
		hst_A[i] = (float)i;
		hst_B[i] = (float)(i+1);
	}

	cudaMalloc((void**)&dev_A, size);
	cudaMalloc((void**)&dev_B, size);
	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, hst_B, size, cudaMemcpyHostToDevice);
}

void cleanup(){
	cudaFree(dev_A);
	cudaFree(dev_B);
}