/**
 * @file      main.cpp
 * @brief     Example N-body simulation for CIS 565
 * @authors   Liam Boone, Kai Ninomiya
 * @date      2013-2015
 * @copyright University of Pennsylvania
 */

#include "main.hpp"

// ================
// Configuration
// ================

#define VISUALIZE 1


#define checkCUDAErrorWithLine2(msg) checkCUDAError2(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError2(const char *msg, int line = -1) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		if (line >= 0) {
			fprintf(stderr, "Line %d: ", line);
		}
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}


/**
 * C main function.
 */
int main(int argc, char* argv[]) {
    projectName = "565 CUDA Intro: Matrix_Math";

	
	//on the host
	hst_matrix1 = new float[25];
	hst_matrix2 = new float[25];
	hst_matrix3 = new float[25];

	//on the device
	cudaMalloc((void**)&dev_matrix1, 25 * sizeof(float));
	checkCUDAErrorWithLine2("cudaMalloc dev_matrix1 failed!");

	cudaMalloc((void**)&dev_matrix2, 25 * sizeof(float));
	checkCUDAErrorWithLine2("cudaMalloc dev_matrix2 failed!");

	cudaMalloc((void**)&dev_matrix3, 25 * sizeof(float));
	checkCUDAErrorWithLine2("cudaMalloc dev_matrix3 failed!");

    if (init(argc, argv)) {
        mainLoop();
       
        return 0;
    } else {
        return 1;
    }
}

//-------------------------------
//---------RUNTIME STUFF---------
//-------------------------------

std::string deviceName;
GLFWwindow *window;

/**
 * Initialization of CUDA and GLFW.
 */
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

    // Window setup stuff
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        std::cout
            << "Error: Could not initialize GLFW!"
            << " Perhaps OpenGL 3.3 isn't available?"
            << std::endl;
        return false;
    }
    int width = 1280;
    int height = 720;

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, deviceName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize drawing state
    //initVAO();

    // Default to device ID 0. If you have more than one GPU and want to test a non-default one,
    // change the device ID.
    cudaGLSetGLDevice(0);

    cudaGLRegisterBufferObject(planetVBO);
    // Initialize Matrix Math
	//Matrix_Math::initlization();

    projection = glm::perspective(fovy, float(width) / float(height), zNear, zFar);
    glm::mat4 view = glm::lookAt(cameraPosition, glm::vec3(0), glm::vec3(0, 0, 1));

    projection = projection * view;

    //initShaders(program);

    glEnable(GL_DEPTH_TEST);

    return true;
}




//====================================
// Main loop
//====================================
void mainLoop()
{
	//Matrix_Math::initlization();
	Matrix_Math::GetInputMatrix();
	Matrix_Math::MatrixAddOnDevice();
	Matrix_Math::PrintResult();
	Matrix_Math::MatrixSubOnDevice();
	Matrix_Math::PrintResult();
	Matrix_Math::MatrixMulOnDevice();
	Matrix_Math::PrintResult();
	Matrix_Math::Terminate();
}


void errorCallback(int error, const char *description) {
	fprintf(stderr, "error %d: %s\n", error, description);
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GL_TRUE);
	}
}



