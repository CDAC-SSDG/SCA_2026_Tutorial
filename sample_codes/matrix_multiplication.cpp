#include <sycl.hpp>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace paras;

#define N 4

int main() {
    sycl::queue q1(sycl::default_selector{});
    cout << "Running on device: " << q1.get_device().get_info<sycl::info::device::name>() << "\n";

    srand(time(0)); // Seed for random number generation

    int A[N][N], B[N][N], C[N][N];

    // Initialize matrices A and B with random values
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10; // Random integers between 0 and 9
            B[i][j] = rand() % 10;
        }

    // Create SYCL buffers
    sycl::buffer<int, 2> bufferA((int*)A, sycl::range<2>(N, N));
    sycl::buffer<int, 2> bufferB((int*)B, sycl::range<2>(N, N));
    sycl::buffer<int, 2> bufferC(sycl::range<2>(N, N));

    // Submit matrix multiplication kernel
    q1.submit([&](sycl::handler& cgh) {
        auto a = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto b = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto result = bufferC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class MatrixMultiply>(sycl::range<2>(N, N), [=](sycl::item<2> item) {
            int i = item.get_id(0);
            int j = item.get_id(1);
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += a[i][k] * b[k][j];

            result[item] = sum;
        });
    });

    q1.wait();

    // Copy result from bufferC to host array C
    {
        sycl::host_accessor result(bufferC, sycl::read_only);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i][j] = result[i][j];
    }

    // Print matrices
    cout << "\n\nFirst matrix (A):\n";
    for (int i = 0; i < N; i++) {
        cout << "\t\t\t";
        for (int j = 0; j < N; j++)
            cout << A[i][j] << " ";
        cout << "\n";
    }

    cout << "\nSecond matrix (B):\n";
    for (int i = 0; i < N; i++) {
        cout << "\t\t\t";
        for (int j = 0; j < N; j++)
            cout << B[i][j] << " ";
        cout << "\n";
    }

    cout << "\nResultant matrix (C = A x B):\n";
    for (int i = 0; i < N; i++) {
        cout << "\t\t\t";
        for (int j = 0; j < N; j++)
            cout << C[i][j] << " ";
        cout << "\n";
    }

    return 0;
}

