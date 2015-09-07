#include "matrix_math.h"

float *dev_mat_A;
float *dev_mat_B;
float *dev_mat_C;

cudaEvent_t start, stop;

void CUDA_matrix_math::initialize() {
	cudaMalloc((void**)&dev_mat_A, sizeof(float) * 25);
	cudaMalloc((void**)&dev_mat_B, sizeof(float) * 25);
	cudaMalloc((void**)&dev_mat_C, sizeof(float) * 25);
}

void CUDA_matrix_math::teardown() {
	cudaFree(dev_mat_A);
	cudaFree(dev_mat_B);
	cudaFree(dev_mat_C);
}

__global__ void mat_add(float *A, float *B, float *C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i > 4 || j > 4) return;
	int index = i + j * 5;
	C[index] = A[index] + B[index];
}

__global__ void mat_sub(float *A, float *B, float *C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i > 4 || j > 4) return;
	int index = i + j * 5;
	C[index] = A[index] - B[index];
}

__global__ void mat_mul(float *A, float *B, float *C) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i > 4 || j > 4) return;
	int index = i + j * 5;
	float dot_product = 0.0f;

	// 0  1  2  3  4
	// 5  6  7  8  9
	// 10 11 12 13 14
	// 15 16 17 18 19
	// 20 21 22 23 24
	int col_index = index % 5;
	int row_index = (index / 5) * 5;
	for (int k = 0; k < 5; k++) {
		dot_product += A[row_index] * B[col_index];
		col_index += 5;
		row_index += 1;
	}

	C[index] = dot_product;
}

static void setup_timer_events() {
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}

static float teardown_timer_events() {
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return milliseconds;
}

void CUDA_matrix_math::cuda_mat_add(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_A, A, 25 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_B, B, 25 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(5, 5);
	dim3 dimGrid(1, 1);

	setup_timer_events();
	cudaEventRecord(start);
	mat_add <<<dimGrid, dimBlock >>>(dev_mat_A, dev_mat_B, dev_mat_C);
	cudaEventRecord(stop);

	float time = teardown_timer_events();
	printf("Addition operation took about %f milliseconds.\n", time);

	cudaMemcpy(C, dev_mat_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);
}

void CUDA_matrix_math::cuda_mat_sub(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_A, A, 25 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_B, B, 25 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(5, 5);
	dim3 dimGrid(1, 1);

	setup_timer_events();
	cudaEventRecord(start);
	mat_sub <<<dimGrid, dimBlock >>>(dev_mat_A, dev_mat_B, dev_mat_C);
	cudaEventRecord(stop);

	float time = teardown_timer_events();
	printf("Subtraction operation took about %f milliseconds.\n", time);

	cudaMemcpy(C, dev_mat_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);
}

void CUDA_matrix_math::cuda_mat_mul(float *A, float *B, float *C) {
	cudaMemcpy(dev_mat_A, A, 25 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat_B, B, 25 * sizeof(float), cudaMemcpyHostToDevice);

	dim3 dimBlock(5, 5);
	dim3 dimGrid(1, 1);

	setup_timer_events();
	cudaEventRecord(start);
	mat_mul <<<dimGrid, dimBlock >>>(dev_mat_A, dev_mat_B, dev_mat_C);
	cudaEventRecord(stop);

	float time = teardown_timer_events();
	printf("Multiplication operation took about %f milliseconds.\n", time);
	cudaMemcpy(C, dev_mat_C, 25 * sizeof(float), cudaMemcpyDeviceToHost);
}