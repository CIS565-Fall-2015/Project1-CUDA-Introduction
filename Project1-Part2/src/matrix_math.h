#pragma once
#include<cuda.h>

namespace Matrix_Math{
	float add(int inputSize,int blockSize,float *A,float *B,float *C);
	float sub(int inputSize,int blockSize,float *A,float *B,float *C);
	float mul(int inputSize,int blockSize,float *A,float *B,float *C);
	void initiate(int size,float *A,float *B);
	void copyBack(int size,float *C);
}