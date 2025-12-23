# Module 3: Memory Management Using Unified Shared Memory (USM)

In this module, we introduce **Unified Shared Memory (USM)** as a flexible and intuitive memory management model in SYCL. The objective of this module is to help participants understand how memory is allocated, accessed, and shared between the host and devices in heterogeneous systems.

We begin by discussing the challenges of explicit data movement in accelerator programming and how USM simplifies this process by providing a **pointer-based memory model** similar to traditional C and C++ programming. Participants will learn about the different types of USM allocations—**host**, **device**, and **shared**—and the access guarantees provided by each type.

This module explains how USM enables direct use of pointers inside SYCL kernels, reducing the complexity associated with buffer–accessor based memory management. 

By the end of this module, participants will understand when and how to use USM effectively, the trade-offs between different USM allocation types, and how USM helps achieve both **programming simplicity and high performance** in SYCL applications.

