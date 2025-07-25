#include <sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <fstream>

using namespace paras;

int main() {
    size_t n;

    std::cout << "============================================\n";
    std::cout << "       APPROXIMATION OF PI USING SYCL       \n";
    std::cout << "============================================\n";
    std::cout << "Enter number of intervals: ";
    std::cin >> n;

    if (n <= 0) {
        std::cerr << "Error: Number of intervals must be greater than 0.\n";
        return EXIT_FAILURE;
    }

    const double dx = 1.0 / n;

    sycl::queue q(sycl::default_selector{});
    std::cout << "\nSYCL Device Selected:\n>> "
              << q.get_device().get_info<sycl::info::device::name>()
              << "\n--------------------------------------------\n";

    std::vector<double> results(n, 0.0);

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
    for (const auto& val : results)
        pi += val;
    pi *= 4.0;

    // Write result to file
    std::string outputFile = "pi_output.dat";
    std::ofstream out(outputFile);
    if (out.is_open()) {
        out << "============================================\n";
        out << "       APPROXIMATION OF PI USING SYCL       \n";
        out << "============================================\n";
        out << "Number of Intervals : " << n << "\n";
        out << std::fixed << std::setprecision(10)
            << "Approximated Pi     : " << pi << "\n";
        out << "============================================\n";
        out.close();

        std::cout << "Output successfully stored in: " << outputFile << "\n";
    } else {
        std::cerr << "Error: Unable to write to file.\n";
    }

    return 0;
}

