#include <iostream>
#include <vector>
#include <sycl.hpp>

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

    // Initialize vectors
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    float result = 0.0f;

    // Print input vectors
    std::cout << "Vector A: [ ";
    for (int i = 0; i < N; ++i) std::cout << a[i] << " ";
    std::cout << "]\n";

    std::cout << "Vector B: [ ";
    for (int i = 0; i < N; ++i) std::cout << b[i] << " ";
    std::cout << "]\n------------------------------\n";

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

    // Output result
    auto host_result = result_buf.get_access<sycl::access::mode::read>();
    std::cout << "Dot Product Result: " << host_result[0] << std::endl;
    std::cout << "==============================\n";

    return 0;
}

