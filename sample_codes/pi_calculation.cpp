#include <sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>  // for std::setprecision

using namespace std;
using namespace paras;

int main() {
    size_t n;

    // Banner
    cout << "============================================\n";
    cout << "       APPROXIMATION OF PI USING SYCL       \n";
    cout << "============================================\n";
    cout << "Enter number of intervals: ";
    cin >> n;

    const double dx = 1.0 / n;

    sycl::queue q(sycl::default_selector{});
    cout << "\nSYCL Device Selected:\n>> "
         << q.get_device().get_info<sycl::info::device::name>()
         << "\n--------------------------------------------\n";

    vector<double> results(n, 0.0);

    {
        sycl::buffer<double, 1> result_buf(results.data(), sycl::range<1>(n));

        q.submit([&](sycl::handler& h) {
            auto result_acc = result_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for<class ComputePi>(sycl::range<1>(n), [=](sycl::id<1> i) {
                double x = (i[0] + 0.5) * dx;
                result_acc[i] = std::sqrt(1.0 - x * x) * dx;
            });
        });
    }

    double pi = 0.0;
    for (const auto& val : results) {
        pi += val;
    }
    pi *= 4.0;

    std::cout << std::fixed << std::setprecision(10)
              << "Approximation of Pi: " << pi << "\n";
    std::cout << "============================================\n";

    return 0;
}

