#include <cuda.h>
#include <stdio.h>

#define blockSize 512.0

__global__ void mat_add(float* d_A, float* d_B, float* d_C, int dim){
	int ptr = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(ptr < dim * dim){
		d_C[ptr] = d_A[ptr] + d_B[ptr];
	}
}

__global__ void mat_sub(float* d_A, float* d_B, float* d_C, int dim){
	int ptr = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(ptr < dim * dim){
		d_C[ptr] = d_A[ptr] - d_B[ptr];
	}
}

__global__ void mat_mul(float* d_A, float* d_B, float* d_C, int dim){
	int ptr = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(ptr < dim * dim){
		int row = ptr / dim;
		int col = ptr % dim;

		float value = 0;

		for (int i = 0; i < dim; i++)
			value += d_A[(row * dim) + i] * d_B[(i * dim) + col];

		d_C[ptr] = value;
	}
}

void checkCudaError(){
	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess)
		printf("%s\n", cudaGetErrorString(err));
}

void printMatrix(float* mat, int dim) {
	int ptr = 0;
	for(int row = 0; row < dim; row++){
		for(int col = 0; col < dim; col++){
			ptr = col + (row * dim);
			printf("%5.2f\t", mat[ptr]);
		}
		printf("\n");
	}
}

void runCPU(float* h_C, float* h_A, float* h_B, int dim){
	//mult test
	int max = dim * dim;
	for(int ptr = 0; ptr < max; ptr++){
		int row = ptr / dim;
		int col = ptr % dim;

		float value = 0;

		for (int i = 0; i < dim; i++)
			value += h_A[(row * dim) + i] * h_B[(i * dim) + col];

		h_C[ptr] = value;
	}
}

void runCUDA(float** C, float* A, float* B, int dim){
	float *d_A, *d_B, *d_C;
	int size = (dim * dim) * sizeof(float);

	cudaMalloc((void**)&d_A, size);
	cudaMalloc((void**)&d_B, size);
	cudaMalloc((void**)&d_C, size);

	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	//TODO: perform a calculation here
	int numBlock = ceil(dim * dim / blockSize);
	mat_mul<<<numBlock, blockSize>>>(d_A, d_B, d_C, dim);

	cudaMemcpy(*C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main() {

	cudaFree(0); //catches the lazy wakeup

	int dim = 5;
	int count = 1;
	float totalTime = 0;

	float h_A[] = {8, 3, 4, 1, 2,
	              16, 9, 2, 3, 5,
	              4, 4, 0, 1, 3,
	              2, 1, 17, 6, 9,
	              18, 2, 3, 2, 1};

	float h_B[] = {2, 0, 0, 0, 0,
				   0, 2, 0, 0, 0,
				   0, 0, 2, 0, 0,
				   0, 0, 0, 2, 0,
		           0, 0, 0, 0, 2};

	float *h_C = (float*)malloc(dim*dim*sizeof(float));

	for(int i = 0; i < count; i++) {
		/// perfs eval
		cudaEvent_t beginEvent;
		cudaEvent_t endEvent;

		cudaEventCreate( &beginEvent );
		cudaEventCreate( &endEvent );

		cudaEventRecord( beginEvent, 0 );

		runCUDA(&h_C, h_A, h_B, dim);

		cudaEventRecord( endEvent, 0 );
		cudaEventSynchronize( endEvent );

		float tmp;
		cudaEventElapsedTime( &tmp, beginEvent, endEvent );
		totalTime += tmp;
	}

	printf("Time: %.4f ms \n", totalTime/count);
	printMatrix(h_C, dim);

}
