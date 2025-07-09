#include <sycl.hpp>
#include <math.h>
#include <iostream>
#include <vector>

using namespace paras;  

constexpr size_t N = 10;

int main() {
    // Create a SYCL queue 
    sycl::queue myQueue{sycl::default_selector{}};

    std::cout << "Device: " 
              << myQueue.get_device().get_info<sycl::info::device::name>() 
              << std::endl;

    // Host vectors
    std::vector<int> A(N, 1);
    std::vector<int> B(N, 2);
    std::vector<int> C(N, 0);

    // SYCL buffers
    sycl::buffer<int, 1> bufferA(A.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> bufferB(B.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> bufferC(C.data(), sycl::range<1>(N));

    // Submit the kernel
    myQueue.submit([&](sycl::handler& cgh) {
        // Accessors
        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.get_access<sycl::access::mode::write>(cgh);

        // Vector addition kernel
        cgh.parallel_for<class VectorAddition>(
            sycl::range<1>(N), [=](sycl::id<1> index) {
                accessorC[index] = accessorA[index] + accessorB[index];
            });
    });

    // Wait for completion
    myQueue.wait();

    // Read result back on host
    auto D = bufferC.get_access<sycl::access::mode::read>();

    std::cout << "Result after vector addition:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << "C[" << i << "] = " << D[i] << "\n";
    }

    return 0;
}

