/**
* @file      main.cpp
* @brief     Matrix math assignment for CIS 565
* @authors   Bradley Crusco
* @date      2015
*/

#include "main.hpp"

/**
* C main function.
*/
int main(int argc, char* argv[]) {
	fprintf(stdout, "Bradley Crusco - CIS-565 - Matrix Math\n\n");

	// Initialize matrices
	MatrixMath::initialization(5);

	// Execute tests for addition, subtraction, and multiplication
	MatrixMath::run_tests();
}