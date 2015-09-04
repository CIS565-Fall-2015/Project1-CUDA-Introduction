#include "main.hpp"

#define EPSILON 0.0001f

static bool nearlyEqual(float a, float b, float epsilon) {
	const float absA = fabs(a);
	const float absB = fabs(b);
	const float diff = fabs(a - b);

	if (a == b) { // shortcut
		return true;
	}
	else if (a * b == 0) { // a and/or b are zero
		return diff < (epsilon * epsilon);
	}
	else { // use relative error
		return diff / (absA + absB) < epsilon;
	}
}

int main(int argc, char* argv[]) {
	CUDA_matrix_math::initialize();
	float *hst_mat_A = (float*)malloc(sizeof(float) * 25);
	float *hst_mat_B = (float*)malloc(sizeof(float) * 25);
	float *hst_mat_C = (float*)malloc(sizeof(float) * 25);

	for (int i = 0; i < 25; i++) {
		hst_mat_A[i] = i + 1;
		hst_mat_B[i] = i + 1;
	}

	// values to assert against
	float hst_add_check[25];
	for (int i = 0; i < 25; i++) {
		hst_add_check[i] = (i + 1) * 2;
	}

	float hst_sub_check[25];
	for (int i = 0; i < 25; i++) {
		hst_sub_check[i] = 0;
	}

	float hst_mul_check[25] = {
		215.0f, 230.0f, 245.0f, 260.0f, 275.0f,
		490.0f, 530.0f, 570.0f, 610.0f, 650.0f,
		765.0f, 830.0f, 895.0f, 960.0f, 1025.0f,
		1040.0f, 1130.0f, 1220.0f, 1310.0f, 1400.0f,
		1315.0f, 1430.0f, 1545.0f, 1660.0f, 1775.0f
	};

	CUDA_matrix_math::cuda_mat_add(hst_mat_A, hst_mat_B, hst_mat_C);
	for (int i = 0; i < 25; i++) {
		assert(nearlyEqual(hst_mat_C[i], hst_add_check[i], EPSILON));
	}

	CUDA_matrix_math::cuda_mat_sub(hst_mat_A, hst_mat_B, hst_mat_C);
	for (int i = 0; i < 25; i++) {
		assert(nearlyEqual(hst_mat_C[i], hst_sub_check[i], EPSILON));
	}

	CUDA_matrix_math::cuda_mat_mul(hst_mat_A, hst_mat_B, hst_mat_C);
	for (int i = 0; i < 25; i++) {
		assert(nearlyEqual(hst_mat_C[i], hst_mul_check[i], EPSILON));
	}

	free(hst_mat_A);
	free(hst_mat_B);
	free(hst_mat_C);
}