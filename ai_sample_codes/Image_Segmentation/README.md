# README
---
This code demonstrates an image segmentation algorithm using K-Means clustering, implemented in SYCL. It is fully compatible with the ParaS SYCL compiler, enabling efficient compilation and execution across heterogeneous computing platforms. With support for both CPU and GPU targets, it allows developers to exploit device-level parallelism while ensuring performance portability. By utilizing the ParaS compiler, users can seamlessly compile and run this SYCL-based image segmentation application on a range of hardware backends.

## Compilation
This code can be compiled using the ParaS compiler for both CPU and GPU targets as specified below.

### For CPUs

For any CPUs:

parascc image_thresholding_sycl.cpp -o clustering \`pkg-config --cflags --libs opencv4\

### For GPUs

For CUDA Enabled NVIDIA GPU:

parascc image_thresholding_sycl.cpp -o clustering -parasdevice cuda:sm_<x> \`pkg-config --cflags --libs opencv4\

For HIP Enabled AMD GPU:

parascc image_thresholding_sycl.cpp -o clustering -parasdevice hip:gfx<x> \`pkg-config --cflags --libs opencv4\

## Expected Output

**Input 1:**

<img src="input_img1.png" alt="Input 1" width="400"/>

**Output 1:**

<img src="output_img_1.png" alt="Output 1" width="400"/>

**Input 2:**

<img src="input_img_2.png" alt="Input 2" width="400"/>

**Output 2:**

<img src="output_img_2.png" alt="Output 2" width="400"/>
