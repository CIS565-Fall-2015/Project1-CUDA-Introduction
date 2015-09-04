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

#include "matrix_math.hpp"

//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);

void mat_print(float* mat, int width);