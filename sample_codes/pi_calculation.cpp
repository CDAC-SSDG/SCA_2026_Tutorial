#include <sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip> // For setting the precision of the output

using namespace paras; 

int main() {
    // Number of intervals - increase for higher precision
    const size_t n = 100000000; // 100 million intervals
    const double dx = 1.0 / n;

    // Create a queue 
    sycl::queue q(sycl::default_selector{});

    std::cout << "Device: "
              << q.get_device().get_info<sycl::info::device::name>()
              << std::endl;

    // Buffer to store partial results
    std::vector<double> results(n, 0.0);

    {
        sycl::buffer<double, 1> result_buf(results.data(), sycl::range<1>(n));

        // Submit the command group
        q.submit([&](sycl::handler& h) {
            auto result_acc = result_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for<class ComputePi>(sycl::range<1>(n), [=](sycl::id<1> i) {
                double x = (i[0] + 0.5) * dx;
                result_acc[i] = std::sqrt(1.0 - x * x) * dx;
            });
        });
    } // buffer goes out of scope and syncs automatically

    // Accumulate the result
    double pi = 0.0;
    for (const auto& val : results) {
        pi += val;
    }
    pi *= 4.0;

    // Output the approximation with 10 decimal places
    std::cout << std::fixed << std::setprecision(10)
              << "Approximation of Pi: " << pi << std::endl;

    return 0;
}

