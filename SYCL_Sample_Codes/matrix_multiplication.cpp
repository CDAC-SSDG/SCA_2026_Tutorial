#include <iostream>
#include <chrono>
#include <cmath>

#include <sycl.hpp>

using namespace paras::sycl;

// Define PI as a constexpr literal because std::acos(-1.0) is not constexpr
constexpr double PI = 3.14159265358979323846;
#define LINE "--------------------" // A line for fancy output

// Function declarations
void initial_value(queue &queue, const unsigned int n, const double dx, const double length, buffer<double, 2> &u);
void zero(queue &queue, const unsigned int n, buffer<double, 2> &u);
void solve(queue &queue, const unsigned int n, const double alpha, const double dx, const double dt, buffer<double, 2> &u, buffer<double, 2> &u_tmp);
double solution(const double t, const double x, const double y, const double alpha, const double length);
double l2norm(const unsigned int n, const double *u, const int nsteps, const double dt, const double alpha, const double dx, const double length);

int main(int argc, char *argv[]) {

  auto start = std::chrono::high_resolution_clock::now();

  unsigned int n = 6000;
  int nsteps = 2;

  if (argc == 3) {
    n = atoi(argv[1]);
    if (n < 0) {
      std::cerr << "Error: n must be positive" << std::endl;
      exit(EXIT_FAILURE);
    }
    nsteps = atoi(argv[2]);
    if (nsteps < 0) {
      std::cerr << "Error: nsteps must be positive" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  double alpha = 0.1;
  double length = 1000.0;
  double dx = length / (n + 1);
  double dt = 0.5 / nsteps;
  double r = alpha * dt / (dx * dx);

  queue queue{default_selector{}};

  std::cout << std::endl
            << " MMS heat equation" << std::endl
            << std::endl
            << LINE << std::endl
            << "Problem input" << std::endl
            << std::endl
            << " Grid size: " << n << " x " << n << std::endl
            << " Cell width: " << dx << std::endl
            << " Grid length: " << length << "x" << length << std::endl
            << std::endl
            << " Alpha: " << alpha << std::endl
            << std::endl
            << " Steps: " << nsteps << std::endl
            << " Total time: " << dt * (double)nsteps << std::endl
            << " Time step: " << dt << std::endl
            << " SYCL device: " << queue.get_device().get_info<info::device::name>() << std::endl
            << LINE << std::endl;

  std::cout << "Stability" << std::endl << std::endl;
  std::cout << " r value: " << r << std::endl;
  if (r > 0.5)
    std::cout << " Warning: unstable" << std::endl;
  std::cout << LINE << std::endl;

  buffer<double, 2> u{range<2>{n, n}};
  buffer<double, 2> u_tmp{range<2>{n, n}};

  initial_value(queue, n, dx, length, u);
  zero(queue, n, u_tmp);
  queue.wait();

  auto tic = std::chrono::high_resolution_clock::now();
  for (int t = 0; t < nsteps; ++t) {
    solve(queue, n, alpha, dx, dt, u, u_tmp);

    // Swap buffers
    auto tmp = std::move(u);
    u = std::move(u_tmp);
    u_tmp = std::move(tmp);
  }
  queue.wait();
  auto toc = std::chrono::high_resolution_clock::now();

  // Access result on host
  auto u_host = u.get_access<access::mode::read>().get_pointer();

  double norm = l2norm(n, u_host, nsteps, dt, alpha, dx, length);

  auto stop = std::chrono::high_resolution_clock::now();

  std::cout << "Results" << std::endl
            << std::endl
            << "(L2norm): " << norm << std::endl
            << "Solve time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() << std::endl
            << "Total time (s): " << std::chrono::duration_cast<std::chrono::duration<double>>(stop - start).count() << std::endl
            << "Bandwidth (GB/s): " << 1.0E-9 * 2.0 * n * n * nsteps * sizeof(double) / std::chrono::duration_cast<std::chrono::duration<double>>(toc - tic).count() << std::endl
            << LINE << std::endl;

  return 0;
}

// Set initial values using MMS scheme
void initial_value(queue &queue, const unsigned int n, const double dx, const double length, buffer<double, 2> &u) {

  queue.submit([&](handler &cgh) {
    auto ua = u.get_access<access::mode::discard_write>(cgh);

    cgh.parallel_for<class initial_value_kernel>(range<2>{n, n}, [=](id<2> idx) {
      int i = idx[1];
      int j = idx[0];
      double y = dx * (j + 1);
      double x = dx * (i + 1);
      ua[idx] = paras::sycl::sin(PI * x / length) * paras::sycl::sin(PI * y / length);
    });
  });
}

// Zero the buffer
void zero(queue &queue, const unsigned int n, buffer<double, 2> &u) {

  queue.submit([&](handler &cgh) {
    auto ua = u.get_access<access::mode::discard_write>(cgh);

    cgh.parallel_for<class zero_kernel>(range<2>{n, n}, [=](id<2> idx) {
      ua[idx] = 0.0;
    });
  });
}

// Solve one timestep with finite difference 5-point stencil
void solve(queue &queue, const unsigned int n, const double alpha, const double dx, const double dt, buffer<double, 2> &u_b, buffer<double, 2> &u_tmp_b) {

  const double r = alpha * dt / (dx * dx);
  const double r2 = 1.0 - 4.0 * r;

  queue.submit([&](handler &cgh) {
    auto u_tmp = u_tmp_b.get_access<access::mode::discard_write>(cgh);
    auto u = u_b.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class solve_kernel>(range<2>{n, n}, [=](id<2> idx) {
      size_t j = idx[0];
      size_t i = idx[1];

      u_tmp[j][i] = r2 * u[j][i] +
                    r * ((i < n - 1) ? u[j][i + 1] : 0.0) +
                    r * ((i > 0) ? u[j][i - 1] : 0.0) +
                    r * ((j < n - 1) ? u[j + 1][i] : 0.0) +
                    r * ((j > 0) ? u[j - 1][i] : 0.0);
    });
  });
}

// Analytical manufactured solution
double solution(const double t, const double x, const double y, const double alpha, const double length) {

  return paras::sycl::exp(-2.0 * alpha * PI * PI * t / (length * length)) * paras::sycl::sin(PI * x / length) * paras::sycl::sin(PI * y / length);
}

// Compute L2 norm between computed and analytical solution
double l2norm(const unsigned int n, const double *u, const int nsteps, const double dt, const double alpha, const double dx, const double length) {

  double time = dt * (double)nsteps;
  double norm = 0.0;

  double y = dx;
  for (int j = 0; j < n; ++j) {
    double x = dx;
    for (int i = 0; i < n; ++i) {
      double exact = solution(time, x, y, alpha, length);
      norm += (u[i + j * n] - exact) * (u[i + j * n] - exact);
      x += dx;
    }
    y += dx;
  }
  return std::sqrt(norm);
}

