CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Ziye Zhou
* Tested on: Windows 8.1, i7-4910 @ 2.90GHz 32GB, GTX 880M 8192MB (Alienware)

### (TODO: Your README)

Include screenshots, analysis, etc. (Remember, this is public, so don't put
anything here that you don't want to share with the world.)

### Questions

**Answer these:**

* Parts 1 & 2: How does changing the tile and block sizes affect performance?
  Why?
Ans: Increase the tile and block sizes will increase the performance to a certain amount but will stay the same after a certain threshold. When the computation unit is not enough for the task, increasing the number will increase the performace, but when computation units are more than enough, increasing the number will not affect the performance.

* Part 1: How does changing the number of planets affect performance? Why?
Ans: The more the planets, the worse the performance. Because when adding more planets into the scene, we need more computation.

* Part 2: Without running comparisons of CPU code vs. GPU code, how would you
  expect the performance to compare? Why? What might be the trade-offs?
Ans: If the number of the parallel computation number is small, I think CPU will win in the comparison; while if the parallel computation number is large, I think GPU will win the comparison. The tradeoff lies between the computation and the context switch. When number of computation is small, the bottleneck will be the context switch. If the number of computation is large, the bottleneck will be the computation itself.

## ScreenShot

N-Body Sim
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/n-body_sim.png?raw=true)

Performance Analysis
*Part 1
(1) visualization on
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/performance_analyse_visualization_on.png?raw=true)
(2) visualizaiton off
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/performance_analyse_visualization_off.png?raw=true)
(3) change blockszie to 64
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/performance_analyse_blocksize_64(base%20128).png?raw=true)
(4) change blocksize to 256
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/performance_analyse_blocksize_256(base%20128).png?raw=true)
(5) change planets number to 10000
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/performance_analyse_num_planets_10000(base%205000).png?raw=true)
(6) change planets number to 2000
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/performance_analyse_num_planets_2000(base%205000).png?raw=true)
*Part 2

![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/part2_timer1.png?raw=true)
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/part2_timer2.png?raw=true)
![alt tag](https://github.com/ziyezhou-Jerry/Project1-CUDA-Introduction/blob/master/images/part2_timer3.png?raw=true)
