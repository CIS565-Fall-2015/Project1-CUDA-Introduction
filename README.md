CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

siqi Huang
Tested on: Windows 7, Inter(R) Core(TM) i7-4870 HQ CPU@ 2.5GHz; GeForce GT 750M(GK107) (Personal Computer)

PART I:
The N body simulation is done and the following five images is in the situation where N=5000;
![](images/nbody/nbody5000/nbody1.jpg)
![](images/nbody/nbody5000/nbody2.jpg)
![](images/nbody/nbody5000/nbody3.jpg)
![](images/nbody/nbody5000/nbody4.jpg)
![](images/nbody/nbody5000/nbody5.jpg)
I have also done the simulation when N=1000 and compare their performance:
The performance of N=1000:
![](images/nbody/nbody_compare/nbody1000_1.png)
![](images/nbody/nbody_compare/nbody1000_2.png)
The performance of N=5000:
![](images/nbody/nbody_compare/nbody5000_1.png)
![](images/nbody/nbody_compare/nbody5000_1.png)
The fps of N=1000 is 60, and 40 when N=5000. Although we have enough thread in GPU to assign the task, their difference comes from the pairwise computation when calculating the acceleration. In N=5000, each thread in GPU has 4000 more computation than in N=1000.
The block size change is done in part2.
