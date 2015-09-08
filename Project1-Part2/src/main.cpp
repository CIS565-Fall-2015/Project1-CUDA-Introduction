#include "main.hpp"

/**
 * C main function.
 */
int main(int argc, char* argv[]) {
    projectName = "565 CUDA Intro: Matrix math";

    if (init(argc, argv)) {
        runCUDA();
        return 0;
    } else {
        return 1;
    }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
float *hst_A, *hst_B, *hst_C;

/**
 * Initialization of CUDA and GLFW.
 */
bool init(int argc, char **argv) {
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
    std::cout << deviceName << std::endl;

	//Set the matrices


	return true;
}

//====================================
// Main loop
//====================================
void runCUDA() {
	
	int N = 5;
	int dim = N*N;
	int index = 0;

	float *A = new float[dim];
	float *B = new float[dim];
	float *C = new float[dim];



	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < N; ++j) {
			index = j + i*N;
			A[index] = index;
			B[index] = 1;
			C[index] = 0;
		}
	}

	//Initialize matirces
    matrix_math::initMatrices(5);
	

	matrix_math::mat_operation( A, B, C, matrix_math::MUL);

	for(int i = 0; i < N; ++i) {
		for(int j = 0; j < N; ++j) {

			index = j + i*N;
			std::cout << i << ":" << C[index] << " ";
	
		}
		std::cout << std::endl;
	}
}
