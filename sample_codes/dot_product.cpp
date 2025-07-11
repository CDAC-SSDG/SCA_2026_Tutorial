#include <iostream>
#include <vector>
#include <sycl.hpp>
#include <fstream>
#include <cstdlib>
#include <ctime>

using namespace paras;

int main() {
    int N;
    std::cout << "==============================\n";
    std::cout << "   DOT PRODUCT USING SYCL\n";
    std::cout << "==============================\n";
    std::cout << "Enter size of vectors (N): ";
    std::cin >> N;
    std::cout << "------------------------------\n";

    // Create SYCL queue
    sycl::queue q(sycl::default_selector{});
    std::cout << "SYCL Device Selected:\n>> "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n------------------------------\n";

    // Initialize vectors with random data
    std::srand(static_cast<unsigned>(std::time(0)));
    std::vector<float> a(N), b(N);
    for (int i = 0; i < N; ++i) {
        a[i] = static_cast<float>(std::rand() % 100);
        b[i] = static_cast<float>(std::rand() % 100);
    }

    float result = 0.0f;

    // Create buffers
    sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> result_buf(&result, sycl::range<1>(1));

    // Submit kernel for dot product
    q.submit([&](sycl::handler &cgh) {
        auto ka = a_buf.get_access<sycl::access::mode::read>(cgh);
        auto kb = b_buf.get_access<sycl::access::mode::read>(cgh);
        auto sum_acc = result_buf.get_access<sycl::access::mode::read_write>(cgh);

        cgh.parallel_for(
            sycl::range<1>(N),
            sycl::reduction(sum_acc, sycl::plus<float>()),
            [=](sycl::id<1> idx, auto &sum) {
                sum += ka[idx] * kb[idx];
            });
    }).wait();

    // Save input and result to .dat file
    std::string filename = "dot_product_output.dat";
    std::ofstream outFile(filename);
    if (outFile.is_open()) {
        outFile << "DOT PRODUCT USING SYCL\n";
        outFile << "Vector Size: " << N << "\n\n";

        outFile << "Vector A: [ ";
        for (int i = 0; i < N; ++i) outFile << a[i] << " ";
        outFile << "]\n";

        outFile << "Vector B: [ ";
        for (int i = 0; i < N; ++i) outFile << b[i] << " ";
        outFile << "]\n\n";

        auto host_result = result_buf.get_access<sycl::access::mode::read>();
        outFile << "Dot Product Result: " << host_result[0] << "\n";

        outFile.close();
        std::cout << "Output successfully stored in: " << filename << "\n";
    } else {
        std::cerr << "Error: Unable to write output to file.\n";
    }

    std::cout << "==============================\n";
    return 0;
}

