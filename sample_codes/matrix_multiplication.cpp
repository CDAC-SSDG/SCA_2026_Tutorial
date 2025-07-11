#include <sycl.hpp>
#include <iostream>
#include <vector>
#include <fstream>
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

    if (N <= 0) {
        cerr << "Error: Size must be greater than 0.\n";
        return EXIT_FAILURE;
    }

    sycl::queue q1(sycl::default_selector{});
    cout << "\n>> Running on device: " 
         << q1.get_device().get_info<sycl::info::device::name>() << "\n";

    srand(time(0));

    vector<vector<int>> A(N, vector<int>(N));
    vector<vector<int>> B(N, vector<int>(N));
    vector<vector<int>> C(N, vector<int>(N, 0));

    // Initialize A and B with random values
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            A[i][j] = rand() % 10;
            B[i][j] = rand() % 10;
        }

    // Flatten matrices for buffer use
    vector<int> flatA(N * N), flatB(N * N), flatC(N * N, 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            flatA[i * N + j] = A[i][j];
            flatB[i * N + j] = B[i][j];
        }

    // SYCL buffers
    sycl::buffer<int, 2> bufferA(flatA.data(), sycl::range<2>(N, N));
    sycl::buffer<int, 2> bufferB(flatB.data(), sycl::range<2>(N, N));
    sycl::buffer<int, 2> bufferC(flatC.data(), sycl::range<2>(N, N));

    // SYCL kernel for matrix multiplication
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

    // Copy result to C
    {
        sycl::host_accessor accC(bufferC, sycl::read_only);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                C[i][j] = accC[i][j];
    }

    // Write everything to output file
    string outputFile = "matrix_multiplication_output.dat";
    ofstream out(outputFile);
    if (out.is_open()) {
        out << "==========================================\n";
        out << "   MATRIX MULTIPLICATION USING SYCL\n";
        out << "==========================================\n";
        out << "Matrix Size: " << N << " x " << N << "\n\n";

        out << "Matrix A:\n";
        for (int i = 0; i < N; i++) {
            out << "\t";
            for (int j = 0; j < N; j++)
                out << A[i][j] << " ";
            out << "\n";
        }

        out << "\nMatrix B:\n";
        for (int i = 0; i < N; i++) {
            out << "\t";
            for (int j = 0; j < N; j++)
                out << B[i][j] << " ";
            out << "\n";
        }

        out << "\nMatrix C = A x B:\n";
        for (int i = 0; i < N; i++) {
            out << "\t";
            for (int j = 0; j < N; j++)
                out << C[i][j] << " ";
            out << "\n";
        }

        out << "==========================================\n";
        out.close();
        cout << "Output successfully written to: " << outputFile << "\n";
    } else {
        cerr << "Error writing output to file.\n";
    }

    return 0;
}

