#include "matrix_math.hpp"

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

	mat_add<<<dimGrid, dimBlock>>>(A, B, C, width);
}

void kern_mat_mul(float* A, float* B, float* C, int width){
	dim3 dimBlock(width, width);
	dim3 dimGrid(1, 1);

	mat_add<<<dimGrid, dimBlock>>>(A, B, C, width);
}

void initialize(){
	int width = 2;
	int size = width * width * sizeof(float);

	//hst_A = (float*)malloc(size);
	//hst_B = (float*)malloc(size);

	//hst_mat1 = {0.0f, 1.0f, 2.0f, 3.0f};
	//hst_mat2 = {3.0f, 2.0f, 1.0f, 0.0f};

	//cudaMalloc((void**)&dev_mat1, size);
	//cudaMalloc((void**)&dev_mat2, size);
	//cudaMemcpy(dev_mat1, hst_mat1, size, cudaMemcpyHostToDevice);
	//cudaMemcpy(dev_mat2, hst_mat2, size, cudaMemcpyHostToDevice);
}

void cleanup(){
	//cudaFree(dev_mat1);
	//cudaFree(dev_mat2);
}