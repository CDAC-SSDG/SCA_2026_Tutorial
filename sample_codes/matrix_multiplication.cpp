#include <sycl.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace paras;

int main() {
    int N;
    cout << "==========================================\n";
    cout << "   MATRIX MULTIPLICATION USING SYCL\n";
    cout << "==========================================\n";
    cout << "Enter the size of square matrices (N x N): ";
    cin >> N;

    sycl::queue q1(sycl::default_selector{});
    cout << "\n>> Running on device: " << q1.get_device().get_info<sycl::info::device::name>() << "\n";

    srand(time(0));

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N));

    // Initialize matrices A and B with random values
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    // Flatten matrices for SYCL buffer
    vector<int> flatA(N * N), flatB(N * N), flatC(N * N);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            flatA[i * N + j] = A[i][j];
            flatB[i * N + j] = B[i][j];
        }

    // SYCL buffers
    sycl::buffer<int, 2> bufferA(flatA.data(), sycl::range<2>(N, N));
    sycl::buffer<int, 2> bufferB(flatB.data(), sycl::range<2>(N, N));
    sycl::buffer<int, 2> bufferC(flatC.data(), sycl::range<2>(N, N));

    // Kernel for matrix multiplication
    q1.submit([&](sycl::handler& cgh) {
        auto a = bufferA.get_access<sycl::access::mode::read>(cgh);
        auto b = bufferB.get_access<sycl::access::mode::read>(cgh);
        auto c = bufferC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class MatrixMultiply>(sycl::range<2>(N, N), [=](sycl::item<2> item) {
            int row = item.get_id(0);
            int col = item.get_id(1);
            int sum = 0;
            for (int k = 0; k < N; ++k)
                sum += a[row][k] * b[k][col];
            c[row][col] = sum;
        });
    });

    q1.wait();

    // Copy results back to matrix C
    {
        sycl::host_accessor accC(bufferC, sycl::read_only);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i][j] = accC[i][j];
    }

    // Display matrices
    cout << "\n------------------------------------------\n";
    cout << "Matrix A:\n";
    for (int i = 0; i < N; i++) {
        cout << "\t";
        for (int j = 0; j < N; j++)
            cout << A[i][j] << " ";
        cout << "\n";
    }

    cout << "\nMatrix B:\n";
    for (int i = 0; i < N; i++) {
        cout << "\t";
        for (int j = 0; j < N; j++)
            cout << B[i][j] << " ";
        cout << "\n";
    }

    cout << "\nMatrix C = A x B:\n";
    for (int i = 0; i < N; i++) {
        cout << "\t";
        for (int j = 0; j < N; j++)
            cout << C[i][j] << " ";
        cout << "\n";
    }

    cout << "------------------------------------------\n";

    return 0;
}

