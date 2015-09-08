#include "main.hpp"

int main(int argc, char* argv[]) {
    projectName = "565 CUDA Intro: Matrix Math";

    if (init(argc, argv)) {
        runCUDA();
        Matrix_Math::cleanUp();
		system("pause");
        return 0;
    } else {
        return 1;
    }
}

/**
 * Initialization of CUDA and GLFW.
 */
std::string deviceName;
GLFWwindow *window;
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

float *hst_mat_a;
float *hst_mat_b;
float *hst_mat_c;
int N;


//====================================
// Main loop
//====================================
void runCUDA() {
	N = 5;

	Matrix_Math::initialize(N);
	hst_mat_a = (float*)malloc(N * N * sizeof(float));
	hst_mat_b = (float*)malloc(N * N * sizeof(float));
	hst_mat_c = (float*)malloc(N * N * sizeof(float));

	for(int i = 0; i < N*N; i++) {
		hst_mat_a[i] = i * 1.0f;
		hst_mat_b[i] = i * 2.0f;
	}

	print_mat(hst_mat_a);
	print_mat(hst_mat_b);
	Matrix_Math::kernMatAdd(N, hst_mat_a, hst_mat_b, hst_mat_c);
	print_mat(hst_mat_c);

	Matrix_Math::kernMatSub(N, hst_mat_a, hst_mat_b, hst_mat_c);
	print_mat(hst_mat_c);

	Matrix_Math::kernMatMul(N, hst_mat_a, hst_mat_b, hst_mat_c);
	print_mat(hst_mat_c);

	Matrix_Math::cleanUp();

}

void print_mat(float* mat) {

	for(int i = 0; i < N*N; i++) {
		if(i%N == 0) {
			std::cout << std::endl;
		}
		std::cout << mat[i] <<",";
	}
	std::cout << std::endl;
}