#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>

namespace matrix_math {

enum operation { ADD, SUB, MUL};
void initMatrices(int dim);
void mat_operation(float *hst_A, float *hst_B, float *hst_C, int op);

}
