#include "main.hpp"
#include <cuda.h>

int main(int argc, char* argv[]) {
	projectName = "565 CUDA Intro: Matrix Math";

	initialize();

	// initialize matrices
	int width = 2;
	int size = width*width*sizeof(float);

	float hst_mat1[4] = {0.0f, 1.0f, 2.0f, 3.0f};
	float hst_mat2[4] = {3.0f, 2.0f, 1.0f, 0.0f};
	float* dev_mat1;
	float* dev_mat2;

	// move them to device memory
	cudaMalloc((void**)&dev_mat1, size);
	cudaMalloc((void**)&dev_mat2, size);
	cudaMemcpy(dev_mat1, hst_mat1, size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mat2, hst_mat2, size, cudaMemcpyHostToDevice);
	float* dev_mat3;
	cudaMalloc((void**)&dev_mat3, size);

	// call the kernel function
	kern_mat_add(dev_mat1, dev_mat2, dev_mat3,width);
	cudaThreadSynchronize();

	// move result to host memory
	float* hst_result = (float*)malloc(size);
	cudaMemcpy(hst_result, dev_mat3, size, cudaMemcpyDeviceToHost);

	// display result
	printf("%f, %f, %f, %f\n", hst_result[0], hst_result[1], hst_result[2], hst_result[3]);

	// clean up
	cudaFree(dev_mat1);
	cudaFree(dev_mat2);
	cudaFree(dev_mat3);

	// Let me actually see the output
	while (1){};
	// clear it all
}