#pragma once

#include <iostream>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <cuda_runtime.h>
#include "matrix_math.h"


//====================================
// Main
//====================================

const char *projectName;

int main(int argc, char* argv[]);

//====================================
// Main loop
//====================================
void runCUDA();

//====================================
// Setup/init Stuff
//====================================
bool init(int argc, char **argv);