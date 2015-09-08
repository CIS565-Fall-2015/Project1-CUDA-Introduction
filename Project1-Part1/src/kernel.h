#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>

namespace Nbody {
void initSimulation(int N);
float stepSimulation(float dt);
void copyPlanetsToVBO(float *vbodptr);
}
