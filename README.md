CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Tongbo Sui
* Tested on: Windows 10, i5-3320M @ 2.60GHz 8GB, NVS 5400M 2GB (Personal)

## Screenshots

### Part 1
![](images/part-1.png)

## Performance Analysis
#### Parts 1 & 2: How does changing the tile and block sizes affect performance? Why?
* Part 1: Changing block sizes (listed below) affects fps performance. However the impact is very little. The grid size is automatically calculated on 5000 planets, hence 40 blocks for block size 128. The minor one from block size 64 to 128 might be due to an increased size in shared memory, and a reduced overhead of block scheduling. After that, utilization of GPU cores might drop down, and each used core has more workload, hence the slight performance drop.
   * 64: 20 fps
   * 128: 22 fps
   * 256: 21.5 fps
   * 512: 20.5 fps
* Part 2: Changing block sizes do have a noticeable impact on the performance, in term of execution time. Operations are tested using two matrices `I` and `2I`. For matrix addition and subtraction, more blocks with a smaller block size runs faster, and there is no significant different for multiplication. The performance improvement is due to utilizing spare GPU cores for more parallelized computation. The even performance for multiplication is probably due to that memory access overhead has evened out the multi-core improvement.
   * 1 Block, 5x5 Thread: Add: 0.006976, Sub: 0.004736, Mul: 0.007200
	* 5 Block, 5x1 Thread: Add: 0.004704, Sub: 0.004672, Mul: 0.007264 

#### How does changing the number of planets affect performance? Why?
Varying the number of planets can significantly impact the performace in terms of frame rate. With block size of 128, and grid size automatically calculated based on number of planets, 5000 planets yields 22 fps, while 500 planet, which is 1/10 of the original size, yields 300 fps. Further reducing planet population to 50 yields a fps of 400.

With 5000 planets, there are 40 blocks for this kernel call. Meanwhile each thread needs to run a for-loop for 5000 steps. The for-loop could be time-consuming since at each step there are a lot of shared memory access. For 5000+ threads this can make a bottleneck for the performance. Besides, NVS 5400 is a pretty old card, and might not support 40 blocks in parallel. Therefore the scheduling involves postponing and switching between blocks, which can cause a significant overhead.

Reducing workload immediately yields a big improvement in performance. However the improvement diminishes as planet population shrinks. With 50 planets, the population can fit into one block. Therefore its performance is bounded by the best performance of a single block.

#### Without running comparisons of CPU code vs. GPU code, how would you expect the performance to compare? Why? What might be the trade-offs?
I expect GPU code to run much faster in term of execution time. CPU code would need a sequential nested loop that cannot be carried out very well in parallel. This nested loop would have an complexity of n^3, for a square n by n matrix. Meanwhile GPU code can utilize the parallelism, where each thread runs in O(n). If we consider the scheduling overhead to be neglegible, then GPU code will be significantly faster due to this complexity difference.

CPU may have a faster memory access, and GPU doesn't. Therefore the actual performance difference might not be that significant, due to overhead caused by excessive memory access from GPU code. Therefore the trade-off will be that to sacrifice I/O time for faster arithmetic operations.
