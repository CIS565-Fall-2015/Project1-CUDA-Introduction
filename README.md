# CUDA Introduction

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

Terry Sun; Arch Linux, Intel i5-4670, GTX 750

## Part 1: A Basic Nbody Simulation

![](images/nbody.gif)

(2500 planets, 0.5s per step)

### Performance

![](images/nbody_perf_plot.png)

I measured performance by disabling visualization and using `CudaEvent`s to time
the kernel invocations (measuring the time elapsed for both `kernUpdateVelPos`
and `kernUpdateAcc`). The graph shows time elapsed (in ms) to update one frame
at block sizes from 16 to 1024 in steps of 8.

Code for performance measuring can be found on the `performance` branch.

Changing the number of planets, as expected, increases the time elapsed for the
kernels, due to a for-loop in the acceleration calculation (which increases
linearly by the number of total planets in the system. More interestingly, it
also changes the way that performance reacts to block size (see n=4096 in the
above plot).

# Part2: An Even More Basic Matrix Library

This library provides addition, subtraction, and multiplication for square
matrices of arbitrary size.

I expect the actual performance of the GPU kernel for addition and subtraction
to run in constant time and thus to be much faster than the respective CPU
operations, as CPU addition and subtraction are linear operations. However, the
GPU operation involves two memory copies of the data (host to device, device to
host), which are also linear time operations.

However, matrix multiplication is a O(n^{1.5}) operation on a CPU and becomes a
O(n) operation on a GPU (becoming O(3n) after taking into account the 2x memory
copy). So I would expect multiplication to exhibit much better performance on
the GPU for larger matrices.
