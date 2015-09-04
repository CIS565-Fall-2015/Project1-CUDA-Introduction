#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>

extern float* hst_A;
extern float* hst_B;
extern float* dev_A;
extern float* dev_B;

//float* hst_mat1;
//float* hst_mat2;
//float hst_mat1[4] = {0.0f, 1.0f, 2.0f, 3.0f};
//float hst_mat2[4] = {3.0f, 2.0f, 1.0f, 0.0f};
//float* dev_mat1;
//float* dev_mat2;

void kern_mat_add(float* A, float* B, float* C, int width);

void initialize();
void cleanup();