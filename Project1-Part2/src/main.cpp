#pragma once
#include <iostream>
#include <cstdlib>
#include <ctime>
#include "matrix_math.h"
using namespace std;

int main(int argc, char* argv[]) {
	int inputSize,blockSize;
	bool outputEnabled=false;
	char c;
	cout<<"please input matrix length"<<endl;
	cin>>inputSize;
	cout<<"please input block size for GPU block"<<endl;
	cin>>blockSize;
	cout<<"please indicate if show the result.(y/n)"<<endl;
	cin>>c;
	if(c=='y') outputEnabled=true;
	float *A=new float[inputSize*inputSize],*B=new float[inputSize*inputSize],*C=new float[inputSize*inputSize];
	for(int i=0;i<inputSize*inputSize;++i){
		A[i]=1;
		B[i]=1;
	}
	float time1=Matrix_Math::add(inputSize,blockSize,A,B,C);
	cout<<"time elapsed on GPU: "<<time1<<"ms"<<endl;

	if(outputEnabled){
		for(int i=0;i<inputSize*inputSize;++i){
			if(i%inputSize==0) cout<<endl;
			cout<<C[i]<<",";
		}
		cout<<endl;
		cout<<"-----------------"<<endl;
	}

	float time2=Matrix_Math::sub(inputSize,blockSize,A,B,C);
	cout<<"time elapsed on GPU: "<<time2<<"ms"<<endl;
	if(outputEnabled){
		for(int i=0;i<inputSize*inputSize;++i){
			if(i%inputSize==0) cout<<endl;
			cout<<C[i]<<",";
		}
		cout<<endl;
		cout<<"-----------------"<<endl;
	}

	float time3=Matrix_Math::mul(inputSize,inputSize,A,B,C);
	cout<<"time elapsed on GPU: "<<time3<<"ms"<<endl;
	if(outputEnabled){
		for(int i=0;i<inputSize*inputSize;++i){
			if(i%inputSize==0) cout<<endl;
			cout<<C[i]<<",";
		}
	}
	return 1;
}