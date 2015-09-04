CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Bradley Crusco
* Tested on: Windows 10, i7-3770K @ 3.50GHz 16GB, 2 x GTX 980 4096MB (Personal Computer)

**N-Body Performance Analysis**

*How does changing the tile and block size affect performance?*
In the first graph you see that, despite a slight improvement in the 384 and 512 block size ranges, that performance decreases as we increase the block size. The reason for this, I suspect, is that the blocks and threads are optimally computationally saturated in this area, giving the increased performance. After this point though, the blocks and threads are under saturated, resulting in performance decrease as we add additional overhead in the form of new blocks that the simulation is not taking proper advantage of.
*How does changing the number of planets effect performance?*
As can be seen in the second graph, the average duration of the kernUpdateAcc function increases exponentially as the number of planets increases. This result is with the number of blocks set to the default of 128 given in the starter code as a baseline. The reason for this seems fairly obvious, since the N-body simulation is a O(N^2) problem, and without altering our block size to compensate we should expect to see this growth in execution time.

**Matrix Math**
*How does changing the tile and block size affect performance?*
Because the computational requirements are so low with the small matrices we are dealing with, block size does not really matter. In fact, we could reduce the block size to 1 (from 25, as I have it now) and see no noticeable difference, as you see in the graph. For very large matrices however this would not be the case.
*Without running comparisons of CPU code vs. GPU code, how would you expect the performance to compare?*
First, for the addition and subtraction operations, I don't expect there to be much of a difference. Addition and subtraction are O(N) operations (where N is the number of elements in the matrix), and so GPU performance is going to be bottlenecked by memory access, keeping us from taking advantage of the GPU's compute power. Matrix multiplication would also have a runtime of O(N), which would be an improvement over the sequential runtime on the CPU, which I believe is O(N^1.5). This is because there is more computational overhead for the multiplication operation, allowing the GPU to shine where the CPU gets slowed down.
