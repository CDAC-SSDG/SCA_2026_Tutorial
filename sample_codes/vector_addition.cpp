#include <sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>   // for setw()
#include <cstdlib>   // for rand(), srand()
#include <ctime>     // for time()

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

    // Seed random generator
    srand(static_cast<unsigned>(time(0)));

    // Create SYCL queue
    sycl::queue myQueue{sycl::default_selector{}};
    cout << "\nSYCL Device Selected : "
         << myQueue.get_device().get_info<sycl::info::device::name>()
         << "\n--------------------------------------------\n";

    // Host vectors with random values
    vector<int> A(N), B(N), C(N, 0);
    for (size_t i = 0; i < N; ++i) {
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

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

    // Write output to file
    string outputFileName = "vector_addition_output.dat";
    ofstream outFile(outputFileName);
    if (outFile.is_open()) {
        outFile << "============================================\n";
        outFile << "         SYCL VECTOR ADDITION PROGRAM       \n";
        outFile << "============================================\n";
        outFile << "Vector Size     : " << N << "\n";

        outFile << "Input Vector A  : [ ";
        for (size_t i = 0; i < N; ++i)
            outFile << setw(3) << A[i] << " ";
        outFile << "]\n";

        outFile << "Input Vector B  : [ ";
        for (size_t i = 0; i < N; ++i)
            outFile << setw(3) << B[i] << " ";
        outFile << "]\n";

        outFile << "Result Vector   : [ ";
        for (size_t i = 0; i < N; ++i)
            outFile << setw(3) << resultAcc[i] << " ";
        outFile << "]\n";

        outFile << "============================================\n";
        outFile.close();

        // Console notification
        cout << "\nOutput successfully written to: " << outputFileName << "\n";
    } else {
        cerr << "Error: Unable to open output file.\n";
    }

    return 0;
}

