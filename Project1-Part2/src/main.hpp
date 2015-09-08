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
#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include "matrix_math.h"

//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);

void runCUDA();
void test_mat_add();
void test_mat_sub();
void test_mat_mul();
void print_mat(float* mat);
bool init(int argc, char **argv);