CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

* Levi Cai
* Tested on: Windows 7, i7 @ 3.4GHz 16GB, Nvidia NVS 310 (Moore 100C Lab)

Nbody:
![](images/project1_basic.PNG)
![](images/project1_few_more_steps.PNG)

## Performance Analysis
![](images/project1_fps_N.png)
![](images/project1_fps_threads.png)

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
