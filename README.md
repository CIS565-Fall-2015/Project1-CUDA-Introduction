CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* SANCHIT GARG
* Tested on: Mac OSX 10.10.4, i7 @ 2.4 GHz, GT 650M 1GB (Personal Computer)

### SANCHIT GARG: ReadMe

## Part 1: N-body Simulation

In this part, I was supposed to work on the framework build by Kai and complete a few function for the amazing N-Body simulation. I wrote the code to find the accelaration of all the planets and then update their position using forward euler integration. Next, we call the CUDA kernels to update the position of all the particles in parallel.

## ScreenShots

![](images/N-Body Simulation.png)


![](images/N-Body Profile.png)



## Part 2: Matrix Math

In this part, you'll set up a CUDA project with some simple matrix math
functionality. Put this in the `Project1-Part2` directory in your repository.

### 1.1. Create Your Project

You'll need to copy over all of the boilerplate project-related files from
Part 1:

* `cmake/`
* `external/`
* `.cproject`
* `.project`
* `GNUmakefile`
* `CMakeLists.txt`
* `src/CMakeLists.txt`

Next, create empty text files for your main function and CUDA kernels:

* `src/main.cpp`
* `src/matrix_math.h`
* `src/matrix_math.cu`

As you work through the next steps, find and use relevant code from Part 1 to
get the new project set up: includes, error checking, initialization, etc.

### 1.2. Setting Up CUDA Memory

As discussed in class, there are two separate memory spaces: host memory and
device memory. Host memory is accessible by the CPU, while device memory is
accessible by the GPU.

In order to allocate memory on the GPU, we need to use the CUDA library
function `cudaMalloc`. This reserves a portion of the GPU memory and returns a
pointer, like standard `malloc` - but the pointer returned by `cudaMalloc` is
in the GPU memory space and is only accessible from GPU code.

We can copy memory to and from the GPU using `cudaMemcpy`. Like C `memcpy`,
you will need to specify the size of memory that you are copying. But
`cudaMemcpy` has an additional argument - the last argument specifies the
whether the copy is from host to device, device to host, device to device, or
host to host.

* Look up documentation on `cudaMalloc` and `cudaMemcpy` if you need to find
  out how to use them - they're not quite obvious.

In an initialization function in `matrix_math.cu`, initialize two 5x5 matrices
on the host and two on the device. Prefix your variables with `hst_` and
`dev_`, respectively, so you know what kind of pointers they are!
These arrays can each be represented as a 1D array of floats:

`{ A_00, A_01, A_02, A_03, A_04, A_10, A_11, A_12, ... }`

Don't forget to call your initialization function from your `main` function in
`main.cpp`.

### 1.3. Creating CUDA Kernels

Given 5x5 matrices A, B, and C (each represented as above), implement the
following functions as CUDA kernels (`__global__`):

* `mat_add(A, B, C)`: `C` is overwritten with the result of `A + B`
* `mat_sub(A, B, C)`: `C` is overwritten with the result of `A - B`
* `mat_mul(A, B, C)`: `C` is overwritten with the result of `A * B`

You should write some tests to make sure that the results of these operations
are as you expect.

Tips:

* `__global__` and `__device__` functions only have access to memory that is
  stored on the device. Any data that you want to use on the CPU or GPU must
  exist in the right memory space. If you need to move data, you can use
  `cudaMemcpy`.
* The triple angle brackets `<<< >>>` provide parameters to the CUDA kernel
  invocation: tile size, block size, and threads per warp.
* Don't worry if your IDE doesn't understand some CUDA syntax (e.g.
  `__device__` or `<<< >>>`). By default, it may not understand CUDA
  extensions.


## Part 3: Performance Analysis

For this project, we will guide you through your performance analysis with some
basic questions. In the future, you will guide your own performance analysis -
but these simple questions will always be critical to answer. In general, we
want you to go above and beyond the suggested performance investigations and
explore how different aspects of your code impact performance as a whole.

The provided framerate meter (in the window title) will be a useful base
metric, but adding your own `cudaTimer`s, etc., will allow you to do more
fine-grained benchmarking of various parts of your code.

REMEMBER:
* Performance should always be measured relative to some baseline when
  possible. A GPU can make your program faster - but by how much?
* If a change impacts performance, show a comparison. Describe your changes.
* Describe the methodology you are using to benchmark.
* Performance plots are a good thing.

### Questions

* Parts 1 & 2: How does changing the tile and block sizes affect performance?
  Why?
* Part 1: How does changing the number of planets affect performance? Why?
* Part 2: Without running comparisons of CPU code vs. GPU code, how would you
  expect the performance to compare? Why? What might be the trade-offs?

**NOTE: Nsight performance analysis tools *cannot* presently be used on the lab
computers, as they require administrative access.** If you do not have access
to a CUDA-capable computer, the lab computers still allow you to do timing
mesasurements! However, the tools are very useful for performance debugging.


## Part 4: Write-up

1. Update all of the TODOs at the top of this README.
2. Add your performance analysis.


## Submit

If you have modified any of the `CMakeLists.txt` files at all (aside from the
list of `SOURCE_FILES`), you must test that your project can build in Moore
100B/C. Beware of any build issues discussed on the Google Group.

1. Open a GitHub pull request so that we can see that you have finished.
   The title should be "Submission: YOUR NAME".
2. Send an email to the TA (gmail: kainino1+cis565@) with:
   * **Subject**: in the form of `[CIS565] Project 0: PENNKEY`
   * Direct link to your pull request on GitHub
   * In the form of a grade (0-100+), evaluate your own performance on the
     project.
   * Feedback on the project itself, if any.

And you're done!
