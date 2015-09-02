#include "matrix_math.h"

float *hst_mat_A;
float *hst_mat_B;

float *dev_mat_A;
float *dev_mat_B;
float *dev_mat_C;

void CUDA_matrix_math::initialize() {
	hst_mat_A = (float*) malloc(sizeof(float) * 25);
	hst_mat_B = (float*) malloc(sizeof(float) * 25);
	
	cudaMalloc((void**)&dev_mat_A, sizeof(float) * 25);
	cudaMalloc((void**)&dev_mat_B, sizeof(float) * 25);
	cudaMalloc((void**)&dev_mat_C, sizeof(float) * 25);
}

void CUDA_matrix_math::teardown() {
	free(hst_mat_A);
	free(hst_mat_B);

	cudaFree(dev_mat_A);
	cudaFree(dev_mat_B);
	cudaFree(dev_mat_C);
}

__global__ void mat_add(float *A, float *B, float *C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int index = i + j * 5;
	C[index] = A[index] + B[index];
}

void CUDA_matrix_math::cuda_mat_add(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_A, A, 25 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_B, B, 25 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(5, 5);
	dim3 dimGrid(1, 1);
	mat_add <<<dimGrid, dimBlock >>>(dev_mat_A, dev_mat_B, dev_mat_C);

	cudaMemcpy(C, dev_mat_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);
}