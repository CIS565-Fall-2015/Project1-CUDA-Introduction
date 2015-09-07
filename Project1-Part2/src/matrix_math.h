#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <fstream>
#include <iostream>

extern float hst_A[], hst_B[];
extern float *hst_C;

namespace Matrix_Math {
	void init();
	float add(float*, float*, float*);
	float sub(float*, float*, float*);
	float mul(float*, float*, float*);
}
