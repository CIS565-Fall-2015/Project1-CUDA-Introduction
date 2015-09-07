CUDA Introduction
=================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 1**

Terry Sun; Arch Linux, Intel i5-4670, GTX 750

Part1 contains a basic nbody simulation:

![](images/nbody.gif)

(2500 planets, 0.5s per step)

Part2 contains an even more basic matrix math library that provides addition,
subtraction, and multiplication.

## TODO
- [ ] write tests for matrix operations
- [ ] performance analysis
- [ ] respond to questions

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

For Part 1, there are two ways to measure performance:
* Disable visualization so that the framerate reported will be for the the
  simulation only, and not be limited to 60 fps. This way, the framerate
  reported in the window title will be useful.
  * Change `#define VISUALIZE` to `0`.
* For tighter timing measurement, you can use CUDA events to measure just the
  simulation CUDA kernel. Info on this can be found online easily. You will
  probably have to average over several simulation steps, similar to the way
  FPS is currently calculated.

**Answer these:**

* Parts 1 & 2: How does changing the tile and block sizes affect performance?
  Why?
* Part 1: How does changing the number of planets affect performance? Why?
* Part 2: Without running comparisons of CPU code vs. GPU code, how would you
  expect the performance to compare? Why? What might be the trade-offs?
