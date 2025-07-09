#include <iostream>
#include <chrono>
#include <cmath>
#include <sycl.hpp>

using namespace std;
using namespace paras;

#define PI paras::sycl::acos(-1.0)

// Initialize mesh using MMS function
void initial_value(sycl::queue &queue, const unsigned int n, const double dx, const double length, sycl::buffer<double,2>& u) {
    queue.submit([&](sycl::handler& cgh) {
        auto ua = u.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<class initial_value_kernel>(sycl::range<2>{n, n}, [=](sycl::id<2> idx) {
            int i = idx[1], j = idx[0];
            double x = dx * (i + 1);
            double y = dx * (j + 1);
            ua[idx] = paras::sycl::sin(PI * x / length) * paras::sycl::sin(PI * y / length);
        });
    });
}

// Zero out the mesh
void zero(sycl::queue &queue, const unsigned int n, sycl::buffer<double,2>& u) {
    queue.submit([&](sycl::handler& cgh) {
        auto ua = u.get_access<sycl::access::mode::discard_write>(cgh);
        cgh.parallel_for<class zero_kernel>(sycl::range<2>{n, n}, [=](sycl::id<2> idx) {
            ua[idx] = 0.0;
        });
    });
}

// Solve heat equation step
void solve(sycl::queue &queue, const unsigned int n, const double alpha, const double dx, const double dt, sycl::buffer<double,2>& u_b, sycl::buffer<double,2>& u_tmp_b) {
    const double r = alpha * dt / (dx * dx);
    const double r2 = 1.0 - 4.0 * r;

    queue.submit([&](sycl::handler& cgh) {
        auto u_tmp = u_tmp_b.get_access<sycl::access::mode::discard_write>(cgh);
        auto u = u_b.get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for<class solve_kernel>(sycl::range<2>{n, n}, [=](sycl::id<2> idx) {
            size_t j = idx[0], i = idx[1];
            u_tmp[j][i] = r2 * u[j][i]
                        + r * ((i < n-1) ? u[j][i+1] : 0.0)
                        + r * ((i > 0)   ? u[j][i-1] : 0.0)
                        + r * ((j < n-1) ? u[j+1][i] : 0.0)
                        + r * ((j > 0)   ? u[j-1][i] : 0.0);
        });
    });
}

// Analytical solution for L2 norm comparison
double solution(const double t, const double x, const double y, const double alpha, const double length) {
    return paras::sycl::exp(-2.0 * alpha * PI * PI * t / (length * length))
         * paras::sycl::sin(PI * x / length)
         * paras::sycl::sin(PI * y / length);
}

// Compute L2 error norm
double l2norm(const unsigned int n, const double *u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {
    double time = dt * (double)nsteps;
    double norm = 0.0, x = dx, y = dx;

    for (int j = 0; j < n; ++j) {
        x = dx;
        for (int i = 0; i < n; ++i) {
            double exact = solution(time, x, y, alpha, length);
            norm += (u[i + j * n] - exact) * (u[i + j * n] - exact);
            x += dx;
        }
        y += dx;
    }

    return paras::sycl::sqrt(norm);
}

// Main driver function
int main() {
    auto start = chrono::high_resolution_clock::now();

    unsigned int n;
    int nsteps;

    cout << "============================================\n";
    cout << "         HEAT EQUATION SOLVER (SYCL)        \n";
    cout << "============================================\n";
    cout << "Enter grid size (e.g., 6000): ";
    cin >> n;
    if (n <= 0) { cerr << "Error: Grid size must be positive.\n"; return 1; }

    cout << "Enter number of time steps (e.g., 20): ";
    cin >> nsteps;
    if (nsteps <= 0) { cerr << "Error: Time steps must be positive.\n"; return 1; }

    double alpha = 0.1, length = 1000.0;
    double dx = length / (n + 1);
    double dt = 0.5 / nsteps;
    double r = alpha * dt / (dx * dx);

    sycl::queue queue{sycl::default_selector{}};

    // Print configuration
    cout << "\n============== Problem Configuration ==============\n";
    cout << " Grid Size      : " << n << " x " << n << "\n";
    cout << " Domain Length  : " << length << " x " << length << "\n";
    cout << " Cell Width (dx): " << dx << "\n";
    cout << " Alpha          : " << alpha << "\n";
    cout << " Time Steps     : " << nsteps << "\n";
    cout << " Time Step (dt) : " << dt << "\n";
    cout << " Total Time     : " << dt * nsteps << "\n";
    cout << " r (stability)  : " << r << (r > 0.5 ? " [UNSTABLE]" : " [STABLE]") << "\n";
    cout << " SYCL Device    : " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    cout << "===================================================\n";

    // Initialize buffers
    sycl::buffer<double, 2> u{sycl::range<2>{n, n}};
    sycl::buffer<double, 2> u_tmp{sycl::range<2>{n, n}};

    initial_value(queue, n, dx, length, u);
    zero(queue, n, u_tmp);
    queue.wait();

    auto tic = chrono::high_resolution_clock::now();
    for (int t = 0; t < nsteps; ++t) {
        solve(queue, n, alpha, dx, dt, u, u_tmp);
        std::swap(u, u_tmp);
    }
    queue.wait();
    auto toc = chrono::high_resolution_clock::now();

    double* u_host = u.get_access<sycl::access::mode::read>().get_pointer();
    double norm = l2norm(n, u_host, nsteps, dt, alpha, dx, length);
    auto stop = chrono::high_resolution_clock::now();

    double solve_time = chrono::duration<double>(toc - tic).count();
    double total_time = chrono::duration<double>(stop - start).count();

    cout << "\n=================== Results ===================\n";
    cout << " L2 Norm (Error)     : " << norm << "\n";
    cout << " Solve Time (s)      : " << solve_time << "\n";
    cout << " Total Time (s)      : " << total_time << "\n";
    cout << " Bandwidth (GB/s)    : "
         << 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / solve_time << "\n";
    cout << "================================================\n";

    return 0;
}

