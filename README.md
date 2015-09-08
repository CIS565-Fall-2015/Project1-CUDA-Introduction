CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Nada Ouf
* Tested on: Windows 7, i7-2649M @ 2.80GHz 8GB, GTX 520 1024MB

#Part1 Screen Capture
![](images/screen)

# Perfomance Analysis

## Parts 1 & 2: How does changing the grid and block sizes affect performance? Why?

##Analysis Results for part 1 and 2
This analysis was done with N = 500

### Part 1:
Timing for the two kernal functions (update accelaration, update velocity and position)

![](images/part1-blockSize-analysis)

### Part 2:
Timing for the matrix multiplication function

![](images/part2-blockSize-analysis)

For both parts my expectation was that by increasing the block size the time taken by the kernal functions would decrease. This effect is seen when increasing the block size from 32 to 128, however, when the block size is increased further performance doesn't improve.
For a small block size (thread per block) more blocks will be needed for the same problem size N. Changing the block size will improve performance if the number of threads per block makes optimal use of resources.

## Part 1: How does changing the number of planets affect performance? Why?
The number of planets affect performance because it changes the number of blocks per grid. For a constant block size as the number of planets increase the number of blocks will increase.
The change in the number of blocks will affect how resources are divided among the different blocks in the GPU. The number of active bloacks depends on the resources used during computation and as a result will affect scheduling.

## Part 2: Without running comparisons of CPU code vs. GPU code, how would you expect the performance to compare? Why? What might be the trade-offs?
I expect that the time to run of the GPU will be a fraction of that of the CPU since each matrix element is computed in parallel (depending on the number of threads).
The trade-off will be that in case of running the code on the GPU requires copying data from host to device and vice versa which is an expensive operation. 
Cost of copying will be more expensive as the size of the matrix increases.

