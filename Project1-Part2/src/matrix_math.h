#pragma once

#include <cuda.h>
#include <malloc.h>
#include <cuda_runtime.h>

namespace CUDA_matrix_math {
	void initialize();
	void teardown();

	void cuda_mat_add(float *A, float *B, float *C);
}