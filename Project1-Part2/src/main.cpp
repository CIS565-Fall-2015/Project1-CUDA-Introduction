#include<iostream>
#include"matrix_math.h"
int main(int argc, char* argv[]) {

	float A[5][5]; 
	float B[5][5]; 
	float C[5][5];
	for (int i = 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			A[i][j] = 1;
			B[i][j] = 2;
			C[i][j] = 0;
		}
	}

	NMatrix::initialization(&A[0][0],&B[0][0]);
	/***********choose the function to test********/
	//NMatrix::mat_add(&A[0][0], &B[0][0], &C[0][0]);//correct
	NMatrix::mat_mul(&A[0][0], &B[0][0], &C[0][0]);
	//NMatrix::mat_sub(&A[0][0], &B[0][0], &C[0][0]);
	/***********************************************/
	std::cout << "Matirx C =";
	for (int i= 0; i < 5; i++){
		for (int j = 0; j < 5; j++){
			std::cout << C[i][j] << ",";
		}
		std::cout << std::endl;
	}
	getchar();
}
