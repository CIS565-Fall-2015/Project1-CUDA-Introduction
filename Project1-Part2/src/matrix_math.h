#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace Matrix_Math {
void initialize(int N);
void cleanUp();

void kernMatAdd(int N, float *hst_mat_a, float *hst_mat_b, float *hst_mat_c);
void kernMatSub(int N, float *hst_mat_a, float *hst_mat_b, float *hst_mat_c);
void kernMatMul(int N, float *hst_mat_a, float *hst_mat_b, float *hst_mat_c);

}
