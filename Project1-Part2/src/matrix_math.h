#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>

namespace MatrixMath {
	void initialization(int mat_width);
	void cleanup();
	void mat_add(float *A, float *B, float *C);
	void mat_sub(float *A, float *B, float *C);
	void mat_mul(float *A, float *B, float *C);
	void print_mat(float *mat, int width);
	void run_tests();
}