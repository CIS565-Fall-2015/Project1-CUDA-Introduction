#pragma once
#include <cmath>
#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h> 

#define MAX 6 
#define blockSizeX 6
#define blockSizeY 6
#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

extern float hst_A[], hst_B[];
extern float *hst_C;

namespace matrix_math {
	void init();
	float add(float *);
	float sub(float *);
	float mul(float *);
	void end();
}
