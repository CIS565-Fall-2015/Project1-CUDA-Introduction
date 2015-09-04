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
#include "matrix_math.h"

const int N = 5;


int main(int argc, char* argv[]) {
	const char *projectName = "565 CUDA Intro: Matrix Math";
	MatMath::initSimulation(N);
	printf("\n Addition \n");
	MatMath::testFunc(0);
	printf("\n Subtraction \n");
	MatMath::testFunc(1);
	printf("\n Multiplication \n");
	MatMath::testFunc(2);

}


