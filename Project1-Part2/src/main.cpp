#include "main.hpp"
#include <cuda.h>

int main(int argc, char* argv[]) {
	projectName = "565 CUDA Intro: Matrix Math";

	// initialize
	int width = 5;
	int size = width*width*sizeof(float);

	initialize(width);

	printf("Matrix multiplication:\n");

	float* dev_C;
	cudaMalloc((void**)&dev_C, size);
	float* hst_C = (float*)malloc(size);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kern_mat_mul(dev_A, dev_B, dev_C, width);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaMemcpy(hst_C, dev_C, size, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	// Display results
	printf("Time(ms): %f\n", ms);

	mat_print(hst_A, width);
	printf("\t*\t\n");
	mat_print(hst_B, width);
	printf("\t=\t\n");
	mat_print(hst_C, width);
	
	// Cleanup
	cleanup();
	cudaFree(dev_C);

	// Let me actually see the output
	while (1){};
}

void mat_print(float* mat, int width){
	for (int i = 0; i < width; i++){
		printf("\t");
		for (int j = 0; j < width; j++){
			printf("%f\t", mat[j+i*width]);
		}
		printf("\n");
	}
	printf("\n");
}