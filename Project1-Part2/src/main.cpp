#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "config.h"
#include "matrix_math.h"

float *hstA;
float *hstB;
float *hstC;

int main() {
    dev_init();
    hst_init(&hstA, &hstB, &hstC);
    return 1;
}
