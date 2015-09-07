#include "matrix_math.h"

using namespace std;
void printMat(float[]);

float hst_A[] = {
	1, 2, 3, 4, 5, 6,
	0, 1, 2, 3, 4, 7,
	9, 2, 3, 4, 5, 8,
	8, 1, 6, 5, 6, 9,
	7, 0, 9, 8, 7, 0,
	6, 5, 4, 3, 2, 1 };

float hst_B[] = {
	1, 0, 0, 0, 0, 1,
	0, 1, 0, 0, 1, 0,
	0, 0, 1, 1, 0, 0,
	0, 0, 1, 1, 0, 0,
	0, 1, 0, 0, 1, 0,
	1, 0, 0, 0, 0, 1 };

void main(){
	cout << "Matrix A:" << endl;
	printMat(hst_A);

	cout << "\nMatrix B:" << endl;
	printMat(hst_B);

	float *hst_C = new float[MAX * MAX];

	cout << "\nA + B:" << endl;
	float timeAdd = matrix_math::add(hst_C);
	printMat(hst_C);

	cout << "\nA - B:" << endl;
	float timeSub = matrix_math::sub(hst_C);
	printMat(hst_C);

	cout << "\nA * B:" << endl;
	float timeMul = matrix_math::mul(hst_C);
	printMat(hst_C);

	cout << "\n(" << blockSizeX << ", " << blockSizeY << ") blocks per grid" << endl;
	cout << "(" << (MAX / blockSizeX) << ", " << (MAX / blockSizeY) << ") threads per block" << endl;
	
	cout << "\nAddition time:   " << timeAdd << " ms" << endl;
	cout << "Subtraction time:   " << timeSub << " ms" << endl;
	cout << "Multiplication time:   " << timeMul << " ms" << endl;
}

void printMat(float mat[]){
	cout.flags(ios::right);
	for (unsigned int i = 0; i < MAX * MAX; i++)
		if ((i + 1) % MAX)
			cout << mat[i] << "  ";
		else
			cout << mat[i] << endl;
}