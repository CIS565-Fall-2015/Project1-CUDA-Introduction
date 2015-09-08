#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "matrix_math.h"

//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);


//====================================
// Setup/init Stuff
//====================================
bool init(int argc, char **argv);
