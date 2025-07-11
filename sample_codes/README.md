# README
---
This directory contains sycl sample codes.
## Compilation

### For Any CPUs

`parascc <program.cpp>`

### For NVIDIA and AMD GPUs

`parascc <program.cpp> -parasdevice cuda:sm_<x>/hip:gxf_<x>`


 [Dot Product](dot_product.cpp)

Description: Compute dot product of two vectors

Sample Output:


## [Matrix Multiplication](matrix_multiplication.cpp)

**Description:** Multiplies two square matrices by taking number of rows as command line arguments

**Sample Output:**



## [Vector Addition](vector_addition.cpp)

**Description:** Adds two vectors and displays the resultant

**Sample Output**




## [PI Calculation](pi_calculation.cpp)

**Description:** Calculates the value of PI using monte-carlo method

**Sample Output**




## Heat Equation

**Description:** Calculate the heat transfer Equation using sycl

**Sample Output**


## Convolution

**Description:** This code computes the convolution using sycl

**Sample Output**



