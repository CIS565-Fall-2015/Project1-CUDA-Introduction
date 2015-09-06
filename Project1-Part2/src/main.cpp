#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "utilityCore.hpp"
#include "glslUtility.hpp"
#include "matrix_math.h"

void printMat(float*mat)
{
	for (int i = 0; i < 25; i++)
	{
		printf(" %.1f,",mat[i]);
		if (i %5  == 4) printf("\n");
	}
	printf("\n");
}

int main(int argc, char* argv[]) {
	float  matA[25] =
	{
		1, 1, 1, 1, 1,//4
		2, 2, 2, 2, 2,//9
		3, 3, 3, 3, 3,//14
		4, 4, 4, 4, 4,
		5, 5, 5, 5, 5
	};
	printf("\ninput Matrix A = \n");
	printMat(matA);
	float matB[25]=
	{
		4, 4, 4, 4, 4,
		3, 3, 3, 3, 3,
		2, 2, 2, 2, 2,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 0
	};
	printf("\ninput Matrix B = \n");
	printMat(matB);
	float matC[25];/* = {
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0
	};*/
	//kint a = -1;
	//std::cin >> a;
	MatrixCalc::mat_mul(matA,matB,matC,5);
	printf("\n multiply output Matrix C = A*B = \n");
	printMat(matC);

	MatrixCalc::mat_add(matA, matB, matC, 5);
	printf("\n multiply output Matrix C = A+B =  \n");
	printMat(matC);

	MatrixCalc::mat_sub(matA, matB, matC, 5);
	printf("\n multiply output Matrix C = A-B = \n");
	printMat(matC);
	system("pause");
}