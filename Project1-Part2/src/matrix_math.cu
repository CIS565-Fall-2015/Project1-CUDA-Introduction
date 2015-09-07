#include "matrix_math.h"

float timeAdd = 0;
float timeSub = 0;
float timeMul = 0;

float *dev_A, *dev_B, *dev_C;
unsigned int size = MAX * MAX * sizeof(float);
void checkCUDAError(const char *, int);

cudaEvent_t start, stop;
dim3 fullBlocksPerGrid(blockSizeX, blockSizeY);
dim3 threadsPerBlock(MAX / blockSizeX, MAX / blockSizeY);

__global__ void mat_add(float* A, float *B, float *C){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = i + j * MAX;

	if (idx < MAX * MAX)
		C[idx] = A[idx] + B[idx];
}

__global__ void mat_sub(float* A, float *B, float *C){
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int idx = i + j * MAX;

	if (idx < MAX * MAX)
		C[idx] = A[idx] - B[idx];
}

__global__ void mat_mul(float* A, float *B, float *C){
	float sum = 0;
	unsigned int corner_x = blockIdx.x*blockDim.x;
	unsigned int corner_y = blockIdx.y*blockDim.y;
	unsigned int idx[] = { corner_x + threadIdx.x, corner_y + threadIdx.y};

	//unsigned int corner_idx = corner_x + corner_y * MAX;
	//unsigned int local_idx = threadIdx.x + threadIdx.y * MAX;

	for (unsigned int k = 0; k < MAX; k++)
		sum += A[idx[1] * MAX + k] * B[idx[0] % MAX + k * MAX];
	C[idx[1] * MAX + idx[0] % MAX] = sum;
}

/* Check for CUDA errors; print and exit if there was a problem.*/
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

void matrix_math::init(){
	cudaMalloc((void**)&dev_A, size);
	checkCUDAErrorWithLine("cudaMalloc dev_A failed!");

	cudaMalloc((void**)&dev_B, size);
	checkCUDAErrorWithLine("cudaMalloc dev_B failed!");

	cudaMalloc((void**)&dev_C, size);
	checkCUDAErrorWithLine("cudaMalloc dev_C failed!");

	cudaMemcpy(dev_A, hst_A, size, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy hst_A to dev_A failed!");

	cudaMemcpy(dev_B, hst_B, size, cudaMemcpyHostToDevice);
	checkCUDAErrorWithLine("cudaMemcpy hst_B to dev_B failed!");
}

float matrix_math::add(float *hst_C){
	init();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_add << <fullBlocksPerGrid, threadsPerBlock >> >(dev_A, dev_B, dev_C);
	cudaEventRecord(stop);

	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeAdd, start, stop);

	end();
	return timeAdd;
}

float matrix_math::sub(float *hst_C){
	init();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_sub << <fullBlocksPerGrid, threadsPerBlock >> >(dev_A, dev_B, dev_C);
	cudaEventRecord(stop);

	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeSub, start, stop);

	end();
	return timeSub;
}

float matrix_math::mul(float *hst_C){
	init();
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_mul << <fullBlocksPerGrid, threadsPerBlock >> >(dev_A, dev_B, dev_C);
	cudaEventRecord(stop);

	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);
	checkCUDAErrorWithLine("cudaMemcpy dev_C to hst_C failed!");

	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&timeMul, start, stop);

	end();
	return timeMul;
}

void matrix_math::end(){
	cudaFree(dev_A);
	cudaFree(dev_B);
	cudaFree(dev_C);
}