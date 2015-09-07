#pragma once

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <iostream>




namespace Matrix_Math
{
	

	void initlization();
	void Terminate();
	void GetInputMatrix(); //from IO or something
	void MatrixAddOnDevice();
	void MatrixSubOnDevice();
	void MatrixMulOnDevice();
	void PrintMatrix(float* m_mat);
	void PrintResult();
	 
}