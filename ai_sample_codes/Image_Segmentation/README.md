# README
---
This code demonstrates an image segmentation algorithm using K-Means clustering, implemented in SYCL. It is fully compatible with the ParaS SYCL compiler, enabling efficient compilation and execution across heterogeneous computing platforms. With support for both CPU and GPU targets, it allows developers to exploit device-level parallelism while ensuring performance portability. By utilizing the ParaS compiler, users can seamlessly compile and run this SYCL-based image segmentation application on a range of hardware backends.

## Compilation
This code can be compiled using the ParaS compiler for both CPU and GPU targets as specified below.

### For CPUs

For any CPUs:

`parascc image_thresholding_sycl.cpp -o clustering \`pkg-config --cflags --libs opencv4\``

### For GPUs

For CUDA Enabled NVIDIA GPU:

`parascc image_thresholding_sycl.cpp -o clustering -parasdevice cuda:sm_<x> \`pkg-config --cflags --libs opencv4\``

For HIP Enabled AMD GPU:

`parascc image_thresholding_sycl.cpp -o clustering -parasdevice hip:gfx<x> \`pkg-config --cflags --libs opencv4\``

## Expected Output

**Input 1:**

![Input-1](input_img1.png)

**Output 1:**

![Output1](output_img_1.png)

**Input 2:**

![Input-2](input_img_2.png)

**Output 2:**

![Output2](output_img_2.png)

