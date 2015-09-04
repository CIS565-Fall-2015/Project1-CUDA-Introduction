#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>

extern float* hst_A;
extern float* hst_B;
extern float* dev_A;
extern float* dev_B;

void kern_mat_add(float* A, float* B, float* C, int width);
void kern_mat_sub(float* A, float* B, float* C, int width);
void kern_mat_mul(float* A, float* B, float* C, int width);

void initialize(int width);
void cleanup();