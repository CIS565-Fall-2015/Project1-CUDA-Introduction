#define GLM_FORCE_CUDA
#include "matrix_math.h"
#include <cuda.h>
#include <glm/glm.hpp>


extern float *hst_matrix1;
extern float *hst_matrix2;
extern float *hst_matrix3;
extern float *dev_matrix1;
extern float *dev_matrix2;
extern float *dev_matrix3;



#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
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





dim3 ThreadsPerBlock(128);
dim3 BlocksPerGrid(1);


void Matrix_Math::initlization()
{
	//on the host
	hst_matrix1 = new float[25];
	hst_matrix2 = new float[25];
	hst_matrix3 = new float[25];

	cudaError_t err = cudaGetLastError();
	//on the device
	cudaMalloc((void**)&dev_matrix1, 25 * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_matrix1 failed!");

	cudaMalloc((void**)&dev_matrix2, 25*sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_matrix2 failed!");

	cudaMalloc((void**)&dev_matrix3, 25 * sizeof(float));
	checkCUDAErrorWithLine("cudaMalloc dev_matrix3 failed!");
	
}

__global__ void kern_mat_add(float* A, float* B, float* C)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < 25)
	{
		C[index] = A[index] + B[index];
	}
}

__global__ void kern_mat_sub(float* A, float* B, float* C)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < 25)
	{
		C[index] = A[index] - B[index];
	}
}

__global__ void kern_mat_mul(float* A, float* B, float* C)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	

	if (index < 25)
	{
		C[index] = 0;

		int row = index / 5;
		int col = index % 5; 

		for (int i = 0; i < 5; i++)
		{
			C[index] += A[row * 5 + i] * B[i * 5 + col];
		}

	}
}

void Matrix_Math::Terminate()
{
	free(hst_matrix1);
	free(hst_matrix2);
	free(hst_matrix3);
	cudaFree(dev_matrix1);
	cudaFree(dev_matrix2);
	cudaFree(dev_matrix3);
}


void Matrix_Math::GetInputMatrix()
{
	memset(hst_matrix1, 0, 25*sizeof(float));
	memset(hst_matrix2, 0, 25 * sizeof(float));
	memset(hst_matrix3, 0, 25 * sizeof(float));

	hst_matrix1[0] = 1;
	hst_matrix2[0] = 2;
	hst_matrix1[6] = 1;
	hst_matrix2[6] = 1;
	hst_matrix1[12] = 1;
	hst_matrix2[12] = 1;
	hst_matrix1[18] = 1;
	hst_matrix2[18] = 1;
	hst_matrix1[24] = 1;
	hst_matrix2[24] = 1;

	

	//copy mem to the device
	cudaMemcpy(dev_matrix1, hst_matrix1, 25 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrix2, hst_matrix2, 25 * sizeof(float), cudaMemcpyHostToDevice);

}

void Matrix_Math::MatrixAddOnDevice()
{
	kern_mat_add << <BlocksPerGrid, ThreadsPerBlock >> > (dev_matrix1,dev_matrix2,dev_matrix3);
	cudaThreadSynchronize();
	//copy mem to the host
	cudaMemcpy(hst_matrix3, dev_matrix3, 25 * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix_Math::MatrixSubOnDevice()
{
	kern_mat_sub << <BlocksPerGrid, ThreadsPerBlock >> > (dev_matrix1, dev_matrix2, dev_matrix3);
	cudaThreadSynchronize();
	//copy mem to the host
	cudaMemcpy(hst_matrix3, dev_matrix3, 25 * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix_Math::MatrixMulOnDevice()
{
	kern_mat_mul << <BlocksPerGrid, ThreadsPerBlock >> > (dev_matrix1, dev_matrix2, dev_matrix3);
	cudaThreadSynchronize();
	//copy mem to the host
	cudaMemcpy(hst_matrix3, dev_matrix3, 25 * sizeof(float), cudaMemcpyDeviceToHost);
}

void Matrix_Math::PrintMatrix(float* m_mat)
{
	for (int i = 0; i<5; i++)
	{
		for (int j = 0; j<5 ; j++)
		{
			std::cout << hst_matrix3[i*5+j] << " ";
		}

		std::cout << std::endl;
	}
}

void Matrix_Math::PrintResult()
{
	PrintMatrix(hst_matrix3);
}


