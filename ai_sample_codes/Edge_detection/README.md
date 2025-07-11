# README
---
This code demonstrates an edge detection algorithm implemented using SYCL. It is fully compatible with the ParaS SYCL compiler, enabling efficient compilation and execution across heterogeneous computing platforms. With support for both CPU and GPU targets, it allows developers to exploit device-level parallelism while ensuring performance portability. By utilizing the ParaS compiler, users can seamlessly compile and run this SYCL-based edge detection application on a range of hardware backends.
## Compilation
This code can be compiled using the ParaS compiler for both CPU and GPU targets as specified below.


### For CPUs

For any CPUs:

`parascc deblur_upsample_sycl.cpp `

### For NVIDIA and AMD GPUs

`parascc deblur_upsample_sycl.cpp -parasdevice [cuda:sm_<x> / hip:gfx<x>]`

> [!NOTE]
> For AMD GPUs use `hip:gfx\<x\>`
>
> For NVIDIA GPUs use `cuda:sm_\<x\>`
>
> where x is the compute capability of GPU device


## Expected Output

**Input 1:**

![Input-1](input1.jpg)

**Output 1:**

![Output1](edge_output1.jpg)

**Input 2:**

![Input-1](input2.png)

**Output 2:**

![Output1](edge_output2.png)

