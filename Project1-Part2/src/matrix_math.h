#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace MatrixCalc
{
	//float *hst_matA;
	//float *hst_matB;

	void initMats(float *hstMatA, float *hstMatB, int matWidth);

	void mat_add(float*A, float*B, float*C,int width);
	void mat_sub(float*A, float*B, float*C, int width);
	void mat_mul(float*A, float*B, float*C, int width);

	void freeMats();
}