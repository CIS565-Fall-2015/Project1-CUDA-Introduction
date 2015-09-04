#include "main.hpp"
#include <cuda.h>

int main(int argc, char* argv[]) {
	projectName = "565 CUDA Intro: Matrix Math";

	initialize();

	// initialize matrices
	int width = 2;
	int size = width*width*sizeof(float);

	float* hst_mat1 = (float*)malloc(size);
	float* hst_mat2 = (float*)malloc(size);

	for (int i = 0; i < width*width; i++){
		hst_mat1[i] = (float)i;
		hst_mat2[i] = (float)(i + 1);
	}

	mat_print(hst_mat1, width);
	mat_print(hst_mat2, width);

	float* dev_mat1;
	float* dev_mat2;

	// move them to device memory
	cudaMalloc((void**)&dev_mat1, size);
	cudaMalloc((void**)&dev_mat2, size);
	cudaMemcpy(dev_mat1, hst_mat1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hst_mat2, size, cudaMemcpyHostToDevice);
	float* dev_mat3;
	cudaMalloc((void**)&dev_mat3, size);

	// call the kernel function and time it
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	kern_mat_mul(dev_mat1, dev_mat2, dev_mat3,width);
	cudaThreadSynchronize();
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	printf("Time(ms): %f\n",ms);

	// move result to host memory
	float* hst_result = (float*)malloc(size);
	cudaMemcpy(hst_result, dev_mat3, size, cudaMemcpyDeviceToHost);

	// display result
	mat_print(hst_result,width);
	//printf("%f, %f, %f, %f\n", hst_result[0], hst_result[1], hst_result[2], hst_result[3]);

	// clean up
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);
	cudaFree(dev_mat3);

	// Let me actually see the output
	while (1){};
	// clear it all
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