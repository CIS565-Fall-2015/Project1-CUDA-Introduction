#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace Matrix_Math {
void initialize();
void cleanUp();

void mat_add(float *A, float *B, float *C);
void mat_sub(float *A, float *B, float *C);
void mat_mul(float *A, float *B, float *C);
}
