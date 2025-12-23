# Module 4: Submitting a Kernel in SYCL

In this module, we explain how computation is expressed and executed in SYCL through **kernel submission**. The objective of this module is to help participants understand how parallel work is defined and dispatched to a selected device using SYCL’s execution model.

We begin by introducing the **queue** as the primary interface through which work is submitted to a device. Participants will learn how a kernel is launched using the `submit` mechanism, where a command group defines the operations to be executed on the device. This submission process is asynchronous, allowing the host application to continue execution while the kernel runs on the device.

This module then introduces the concept of **range**, which defines the total number of parallel work-items to be executed. Each work-item represents an independent instance of the kernel. To uniquely identify each work-item, SYCL provides the **id** abstraction, which allows kernels to determine which portion of data they are responsible for processing.

We further explain the **`parallel_for`** construct, which is used to launch data-parallel kernels across the specified range. Participants will learn how `parallel_for` maps work-items to data elements, enabling simple and scalable expression of parallelism without explicit thread management.

By the end of this module, participants will understand how kernels are submitted in SYCL, how parallel execution is defined using `range` and `id`, and how `parallel_for` enables portable, data-parallel programming across different hardware architectures.
