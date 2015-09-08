//#pragma once

#include "main.hpp"

int main(int argc, char* argv[]) {
    projectName = "565 CUDA : Matrix Math";

    if (init(argc, argv)) {
    	//std::cout<<"everything working";

    	initialize();

    	doMatrixMath();

    	doPerformanceCalculation();

    	cleanup();
        return 0;
    } else {
        return 1;
    }
}

std::string deviceName;

bool init(int argc, char **argv) {
    // Set window title to "Student Name: [SM 2.0] GPU Name"
    cudaDeviceProp deviceProp;
    int gpuDevice = 0;
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (gpuDevice > device_count) {
        std::cout
                << "Error: GPU device number is greater than the number of devices!"
                << " Perhaps a CUDA-capable GPU is not installed?"
                << std::endl;
        return false;
    }
    cudaGetDeviceProperties(&deviceProp, gpuDevice);
    int major = deviceProp.major;
    int minor = deviceProp.minor;

    std::ostringstream ss;
    ss << projectName << " [SM " << major << "." << minor << " " << deviceProp.name << "]";
    deviceName = ss.str();


    return true;
}

