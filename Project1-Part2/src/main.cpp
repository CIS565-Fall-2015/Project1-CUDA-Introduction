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

int main(int argc, char* argv[]) {
	float  matA[25] =
	{
		1, 1, 1, 1, 1,
		2, 2, 2, 2, 2,
		3, 3, 3, 3, 3,
		4, 4, 4, 4, 4,
		5, 5, 5, 5, 5
	};
	float matB[25]=
	{
		4, 4, 4, 4, 4,
		3, 3, 3, 3, 3,
		2, 2, 2, 2, 2,
		1, 1, 1, 1, 1,
		0, 0, 0, 0, 0
	};
	float matC[25];/* = {
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 0
	};*/
	MatrixCalc::mat_mul(matA,matB,matC,5);
	printf("aaa");
}