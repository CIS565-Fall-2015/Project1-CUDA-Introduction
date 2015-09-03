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
#include "matrix_math.h"

//====================================
// GL Stuff
//====================================

GLuint positionLocation = 0;
const char *attributeLocations[] = { "Position" };





int main(int argc, char* argv[]);

