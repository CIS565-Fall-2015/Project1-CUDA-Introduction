
#include "matrix_math.h"
#include <iostream>


//------------------------
//	Matrices
//------------------------

int *hst_matrix1;
int *hst_matrix2;
int *hst_matrix3;

int *dev_matrix1;
int *dev_matrix2;
int *dev_matrix3;


#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

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

void initialize()
{
	hst_matrix1 = new int[25];
	hst_matrix2 = new int[25];
	hst_matrix3 = new int[25];

	for(int i=0; i<25; i++)
	{
		hst_matrix1[i] = i;
		hst_matrix2[i] = i;
	}

	std::cout<<"Matrix A : \n";
	printMatrix(hst_matrix1);
	std::cout<<"\n\nMatrix B : \n";
	printMatrix(hst_matrix2);
	std::cout<<std::endl<<std::endl;

	//Allocate device memory
	cudaMalloc((void**)&dev_matrix1, 25 * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_matrix2, 25 * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	cudaMalloc((void**)&dev_matrix3, 25 * sizeof(int));
	checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

	//copy values to device
	cudaMemcpy(dev_matrix1, hst_matrix1, 25 * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_matrix2, hst_matrix2, 25 * sizeof(int), cudaMemcpyHostToDevice);
}

void doMatrixMath()
{
	//Addition
	doMatrixAdd();
	cudaMemcpy(hst_matrix3, dev_matrix3, 25 * sizeof(int), cudaMemcpyDeviceToHost);
	std::cout<<"A + B = \n";
	printMatrix(hst_matrix3);
	std::cout<<std::endl<<std::endl;

	//Subtraction
	doMatrixSub();
	cudaMemcpy(hst_matrix3, dev_matrix3, 25 * sizeof(int), cudaMemcpyDeviceToHost);
	std::cout<<"A - B = \n";
	printMatrix(hst_matrix3);
	std::cout<<std::endl<<std::endl;

	//Multiplication
	doMatrixMul();
	cudaMemcpy(hst_matrix3, dev_matrix3, 25 * sizeof(int), cudaMemcpyDeviceToHost);
	std::cout<<"A * B = \n";
	printMatrix(hst_matrix3);
	std::cout<<std::endl<<std::endl;
}

void cleanup()
{
	delete(hst_matrix1);
	delete(hst_matrix2);
	delete(hst_matrix3);

	cudaFree(dev_matrix1);
	cudaFree(dev_matrix2);
	cudaFree(dev_matrix3);
}

void printMatrix(int * mat)
{
	for(int i=0; i<5; ++i)
	{
		for(int j=0; j<5; ++j)
		{
			std::cout<<mat[i*5 + j]<<" ";
		}
		std::cout<<std::endl;
	}
}

int *h_data;

__global__ void mat_add(int * A, int * B, int * C)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < 25)
	{
		C[index] = A[index] + B[index];
	}
}

__global__ void mat_sub(int * A, int * B, int * C)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < 25)
	{
		C[index] = A[index] - B[index];
	}
}

__global__ void mat_mul(int * A, int * B, int * C)
{
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if(index < 25)
	{
		int i, j;
		i = index % 5;
		j = index / 5;

		C[i*5 + j] = A[i*5] * B[j] +
					A[i*5 + 1] * B[(1)*5 + j] +
					A[i*5 + 2] * B[(2)*5 + j] +
					A[i*5 + 3] * B[(3)*5 + j] +
					A[i*5 + 4] * B[(4)*5 + j];
	}
}

int gridSize = 10,
	blockSize = 3;

void doMatrixAdd()
{
	mat_add<<<gridSize, blockSize>>>(dev_matrix1, dev_matrix2, dev_matrix3);
}

void doMatrixSub()
{
	mat_sub<<<gridSize, blockSize>>>(dev_matrix1, dev_matrix2, dev_matrix3);
}

void doMatrixMul()
{
	mat_mul<<<gridSize, blockSize>>>(dev_matrix1, dev_matrix2, dev_matrix3);
}

void doPerformanceCalculation()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	mat_add<<<gridSize, blockSize>>>(dev_matrix1, dev_matrix2, dev_matrix3);
	mat_sub<<<gridSize, blockSize>>>(dev_matrix1, dev_matrix2, dev_matrix3);
	mat_mul<<<gridSize, blockSize>>>(dev_matrix1, dev_matrix2, dev_matrix3);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	std::cout<<"Total time in milliseconds : "<<milliseconds<<std::endl;
}
