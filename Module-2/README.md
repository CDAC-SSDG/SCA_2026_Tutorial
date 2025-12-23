# Module 2: Device Discovery in SYCL

In this module, we focus on **device discovery**, which is a fundamental step in executing SYCL applications on heterogeneous systems. The objective of this module is to help participants understand how SYCL identifies and manages different compute devices such as CPUs, GPUs, and other accelerators in a unified manner.

We begin by introducing the concept of a **SYCL device** and explain how devices represent abstract views of physical hardware. Participants will learn how SYCL applications discover available devices at runtime and how developers can select devices either **automatically** or **explicitly**, depending on application requirements. This enables programs to remain portable while still allowing control over where computation is executed.

This module also explains how devices are associated with **queues**, which together form the foundation for submitting work to a selected device. We discuss how a queue acts as the execution interface between the host application and the device, managing kernel execution and memory operations.

By the end of this module, participants will have a clear understanding of how SYCL performs device discovery, how compute devices are selected, and how this abstraction allows applications to adapt seamlessly to diverse hardware environments without changing application logic.

