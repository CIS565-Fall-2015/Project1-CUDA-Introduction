#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace NMatrix {
	void initialization(float *hst_MA, float *hst_MB);
	void mat_add(float  *A, float  *B, float *C);
	void mat_sub(float  *A, float  *B, float *C);
	void mat_mul(float  *A, float  *B, float *C);
	void endMAtrix();
	//void endSimulation();
}