#include "matrix_math.h"
#include <cuda.h>

using namespace std;

void printMatrix(float* mat, int dim) {
	int i = 0;
	for(int row = 0; row < dim; row++){
		for(int col = 0; col < dim; col++){
			i = (row * dim) + col;
			printf("%6.2f\t", mat[i]);
		}
		printf("\n");
	}
}

int main(){

	int dim = 5;

	float hst_A[] = {1, 2, 3, 4, 5,
		              6, 7, 8, 9, 10,
		              11, 12, 13, 14, 15,
		              10, 9, 8, 7, 6,
		              5.00, 4.02, 3.04, 2.06, 1.08};

	float hst_B[] = {10, 9, 8, 7, 6,
    				5.00, 4.02, 3.04, 2.06, 1.08,
            		11, 12, 13, 14, 15,
            		1, 2, 3, 4, 5,
            		6, 7, 8, 9, 10};

	cout << "Matrix A:" << endl;
	printMatrix(hst_A, dim);

	cout << "Matrix B:" << endl;
	printMatrix(hst_B, dim);

	float *hst_C = new float[dim * dim];

	cout << "A + B:" << endl;
	float timeAdd = Matrix_Math::add(hst_A, hst_B, hst_C);
	printMatrix(hst_C, dim);
	printf("Time: %.4f ms \n", timeAdd);

	cout << "A - B:" << endl;
	float timeSub = Matrix_Math::sub(hst_A, hst_B, hst_C);
	printMatrix(hst_C, dim);
	printf("Time: %.4f ms \n", timeSub);

	cout << "A * B:" << endl;
	float timeMul = Matrix_Math::mul(hst_A, hst_B, hst_C);
	printMatrix(hst_C, dim);
	printf("Time: %.4f ms \n", timeMul);

	//for performance evaluation
	int iters = 1000;
	float time = 0;
	for(int i = 0; i<iters; i++ ){
		time = time + Matrix_Math::add(hst_A, hst_B, hst_C) + Matrix_Math::sub(hst_A, hst_B, hst_C) + Matrix_Math::mul(hst_A, hst_B, hst_C);
	}
	printf("Average Time: %.4f ms \n", time/iters);
}
