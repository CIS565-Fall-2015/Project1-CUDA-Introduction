#pragma once

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <glm/gtx/norm.hpp>

namespace Nbody {
void initSimulation(int N);
void stepSimulation(float dt);
void copyPlanetsToVBO(float *vbodptr);

void endSimulation();


}
