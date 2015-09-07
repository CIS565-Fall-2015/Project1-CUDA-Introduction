CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Ratchpak (Dome) Pongmongkol
* Tested on: OSX Yosemite 10.10.5, i7 @ 2.4GHz 16GB, GT 650M 1024MB (rMBP Early 2013)

#  N-body Simulation: Screenshot 

![](images/SG.png)

## Performance Analysis

## Parts 1 & 2: How does changing the tile and block sizes affect performance? Why?
### Part 1:

![](images/01.png)

My assumption : When the block size is small, the tile size increases. One reason I could think of is that the number of concurrent block is limited (resulting in some block having to wait in the queue). This 'delay' disappears when the tile size decreases to the point that or all threads can run concurrently, so the performance after block size = 128 remains the same. It is also possible that this might be caused by block-switching overhead.

### part 2: 

![](images/02.png)
** averaged from 100 times of execution

My assumption: Since the number of threads per block must be a multiple of 32 (which is already larger than dim*dim, or number of concurrent threads). Also, changing the block size will still result as having tile size = 1. This means changing the block size shouldn't affect its performance.

## Part 1: How does changing the number of planets affect performance? Why?

![](images/03.png)

If we plot graphs with kernUpdateAcc and kernUpdateVelPos, it will show that the first one is increasing exponentially as N grows, while the latter is growing linearly. This makes perfect sense since gravity calculation is O(N^2) and velocity/postion updating is O(N) 

## Part 2: Without running comparisons of CPU code vs. GPU code, how would you expect the performance to compare? Why? What might be the trade-offs?
For such a small input like 5x5 matrices, I would say the performance should be roughly the same. 

GPU running time = cudaMemcpy + kernel func time (for each element in mat C)

CPU running time = dim * dim * time spent on each element in mat C

So if the matrix dimension is low enough that the time spent on cudaMemcpy is longer (which might be the case for mul_add and mul_sub), CPU would win this performance contest. Otherwise, GPU will be slightly faster.
