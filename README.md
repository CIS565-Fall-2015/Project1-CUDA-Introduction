CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Shuai Shao (Shrek)
* Tested on: Windows 7, i5-3210M @ 2.50GHz 4.00GB, GeForce GT 640M LE (Personal Laptop)

Screenshot
---------------------

#### Part-1 N-body
![n-body-screen shot](https://github.com/shrekshao/Project1-CUDA-Introduction/blob/master/images/nbody_screenshot.png?raw=true)

(with a random color fragment shader -_-!!!)

Analysis
---------------------
VS Release build
Run in VS2012 IDE Local Debugger
Visualization turned on




+  **BlockSize (num of threads per block)** (wih N = 5000)

| BlockSize  | GridSize | 100 - 200 updates time(ms)|
| ------------- | ------------- | ----------------|
| 64   | 79 | 3563 |
| 128  | 40 | 3248 |
|192   | 27  |3277
| 256  | 20 | 3253 |
| 512  | 10  | 3467 |

There is a sweet spot of block size for a given task. Too small block size will lead to limited number of active threads. If don't taken into account the registers and shared memory, it is good to make the block size big if only the core number allows.

To mention, blockSize should also set to 32n due to commands called in unit of warp which consist of 32 threads. 



+  **N** (with blockSize = 128)

| N| GridSize|100 - 200 updates time(ms)|GPU Utilization
| ------------- | ----------------|----|
| 50   | 1 |1623|<5%
| 500  | 4 |1628|<5%
| 5000 | 40 |3248|\>90%
| 10000| 79  |11913|\>90%

When N drops to some level, the time won't be shorten. For the utilization of GPU can be low. Most time is spent on CPU calling cudaGLMapBufferObject. When N goes up, since the main bottleneck is the kernUpdateAcc function as it will loop through all the planet array. 

N=500, time line
![enter image description here](https://github.com/shrekshao/Project1-CUDA-Introduction/blob/master/images/N500_timeline.png?raw=true)

N=10000, time line
![enter image description here](https://github.com/shrekshao/Project1-CUDA-Introduction/blob/master/images/N10000_timeline.png?raw=true)

At first I expected the increase would be linear but turns out not. I think it is because the number of block is increased, so that some resources like register becomes the limitation. This change can be seen in the analysis tool.
![enter image description here](https://github.com/shrekshao/Project1-CUDA-Introduction/blob/master/images/N500_limit.png?raw=true) ![enter image description here](https://github.com/shrekshao/Project1-CUDA-Introduction/blob/master/images/N10000_limit.png?raw=true)



+ **Answer to this question: Without running comparisons of CPU code vs. GPU code, how would you expect the performance to compare? Why? What might be the trade-offs?**

Since the matrix size is small, I think the GPU code would have no advantage. The naive CPU code may run faster. To achieve concurrent threads, GPU code brings in two memory copy process that the CPU naive for loop approach don't have. Also the inner for loop for one element cannot be avoided. 



+ **Occupancy**

To optimize, generally we want to maximize number of active threads. The major factors influencing occupancy are shared memory usage, register usage, and thread block size.



Summary
--------------

There are lots of information and knowledge to learn (e.g. Compute to global memory access (CGMA) ratio)if I want a detailed and effective analysis. Yet I'm heading for PennApp so there isn't time. Maybe better next time.



