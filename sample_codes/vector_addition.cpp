#include <sycl.hpp>
#include <iostream>
#include <vector>
#include <iomanip>  // for setw()

using namespace std;
using namespace paras;

int main() {
    size_t N;

    
    cout << "============================================\n";
    cout << "         SYCL VECTOR ADDITION PROGRAM       \n";
    cout << "============================================\n";

    // Take vector size input from user
    cout << "Enter the size of vectors: ";
    cin >> N;

    if (N <= 0) {
        cerr << "Error: Vector size must be greater than 0.\n";
        return EXIT_FAILURE;
    }

    // Create SYCL queue
    sycl::queue myQueue{sycl::default_selector{}};
    cout << "\nSYCL Device Selected : "
         << myQueue.get_device().get_info<sycl::info::device::name>()
         << "\n--------------------------------------------\n";

    // Host vectors
    vector<int> A(N, 1);  // All elements = 1
    vector<int> B(N, 2);  // All elements = 2
    vector<int> C(N, 0);  // Result vector

    // Print input vectors
    cout << "\nInput Vector A  : [ ";
    for (size_t i = 0; i < N; ++i)
        cout << setw(2) << A[i] << " ";
    cout << "]\n";

    cout << "Input Vector B  : [ ";
    for (size_t i = 0; i < N; ++i)
        cout << setw(2) << B[i] << " ";
    cout << "]\n";

    // Create SYCL buffers
    sycl::buffer<int, 1> bufferA(A.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> bufferB(B.data(), sycl::range<1>(N));
    sycl::buffer<int, 1> bufferC(C.data(), sycl::range<1>(N));

    // Submit the kernel
    myQueue.submit([&](sycl::handler& cgh) {
        auto accessorA = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto accessorB = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto accessorC = bufferC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class VectorAddition>(
            sycl::range<1>(N), [=](sycl::id<1> idx) {
                accessorC[idx] = accessorA[idx] + accessorB[idx];
            });
    });

    myQueue.wait();  // Ensure kernel execution finishes

    // Access result vector from buffer
    auto resultAcc = bufferC.get_access<sycl::access::mode::read>();

    // Print result vector
    cout << "\nResult Vector   : [ ";
    for (size_t i = 0; i < N; ++i)
        cout << setw(2) << resultAcc[i] << " ";
    cout << "]\n";

    cout << "============================================\n";

    return 0;
}

