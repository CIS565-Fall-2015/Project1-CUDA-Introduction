#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

#include "config.h"
#include "matrix_math.h"

// ==============================================
// Printing
// ==============================================

void printMatrix(float *m) {
    for (int i = 0; i < ARRAY_SIZE; i++) {
        for (int j = 0; j < ARRAY_SIZE; j++) {
            std::cout << m[(i*ARRAY_SIZE)+j];
            std::cout << "\t";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// ==============================================
// State
// ==============================================

float *hstA;
float *hstB;
float *hstC;

// ==============================================
// Main
// ==============================================

int main() {
    dev_init();
    hst_init(&hstA, &hstB, &hstC);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        for (int j = 0; j < ARRAY_SIZE; j++) {
            int index = (i*ARRAY_SIZE) + j;
            hstA[index] = i + (j*2);
            hstB[index] = j + (i*2);
        }
    }
    cudaMul(hstA, hstB, hstC);
    return 1;
}
