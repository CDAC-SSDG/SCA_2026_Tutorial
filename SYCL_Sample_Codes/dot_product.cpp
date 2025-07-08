#include <iostream>
#include <vector>
#include <sycl.hpp>

using namespace paras;

int main() {
    int N = 10;

    // Initialize vectors with values
    std::vector<float> a(N, 1.0f);
    std::vector<float> b(N, 2.0f);
    float result = 0.0f;

    // Create SYCL queue
    sycl::queue q(sycl::default_selector{});
    std::cout << "Device: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Create buffers
    sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(N));
    sycl::buffer<float, 1> result_buf(&result, sycl::range<1>(1));

    // Submit kernel
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

    // Read and print the result
    auto host_result = result_buf.get_access<sycl::access::mode::read>();
    std::cout << "Dot Product Result: " << host_result[0] << std::endl;

    return 0;
}

