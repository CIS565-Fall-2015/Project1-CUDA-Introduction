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

static void printMatrix(float *matrix, int dimm) {
	for (int i = 0; i < dimm; i++) {
		for (int j = 0; j < dimm; j++) {
			int index = j + i * dimm;
			printf("%f ", matrix[index]);
		}
		printf("\n");
	}
}

int main(int argc, char* argv[]) {
	int device_count = 0;
	cudaGetDeviceCount(&device_count);
	int gpuDevice = 0;
	if (gpuDevice > device_count) {
		std::cout
			<< "Error: GPU device number is greater than the number of devices!"
			<< " Perhaps a CUDA-capable GPU is not installed?"
			<< std::endl;
		return false;
	}

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

	printf("Matrix A is \n");
	printMatrix(hst_mat_A, 5);
	printf("\n");

	printf("Matrix B is \n");
	printMatrix(hst_mat_B, 5);
	printf("\n");

	CUDA_matrix_math::cuda_mat_add(hst_mat_A, hst_mat_B, hst_mat_C);
	printf("After performing addition, C is \n");
	printMatrix(hst_mat_C, 5);
	printf("\n");

	for (int i = 0; i < 25; i++) {
		float actual = hst_mat_C[i];
		float expected = hst_add_check[i];
		assert(nearlyEqual(actual, expected, EPSILON));
	}

	CUDA_matrix_math::cuda_mat_sub(hst_mat_A, hst_mat_B, hst_mat_C);
	printf("After performing subtraction, C is \n");
	printMatrix(hst_mat_C, 5);
	printf("\n");

	for (int i = 0; i < 25; i++) {
		float actual = hst_mat_C[i];
		float expected = hst_sub_check[i];
		assert(nearlyEqual(actual, expected, EPSILON));
	}

	CUDA_matrix_math::cuda_mat_mul(hst_mat_A, hst_mat_B, hst_mat_C);
	printf("After performing multiplication, C is \n");
	printMatrix(hst_mat_C, 5);
	printf("\n");

	for (int i = 0; i < 25; i++) {
		float actual = hst_mat_C[i];
		float expected = hst_mul_check[i];
		assert(nearlyEqual(actual, expected, EPSILON));
	}

	CUDA_matrix_math::teardown();
	free(hst_mat_A);
	free(hst_mat_B);
	free(hst_mat_C);

	system("pause");
}