#include <sycl.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace paras;

int main() {
    size_t width, height;

    cout << "============================================\n";
    cout << "  CONVOLUTION USING SYCL\n";
    cout << "============================================\n";
    cout << "Enter width : ";
    cin >> width;
    cout << "Enter height: ";
    cin >> height;
    cout << "--------------------------------------------\n";

    // Initialize image matrix
    vector<int> image(width * height);
    for (size_t i = 0; i < width * height; ++i)
        image[i] = i + 1;

    // Sobel kernel (horizontal edge detection)
    const vector<int> kernel = {
         1,  0, -1,
         2,  0, -2,
         1,  0, -1
    };

    vector<int> result(width * height, 0);

    // SYCL queue
    sycl::queue queue{sycl::default_selector{}};
    cout << "SYCL Device Used: "
         << queue.get_device().get_info<sycl::info::device::name>()
         << "\n--------------------------------------------\n";

    // Buffers
    sycl::buffer<int> imageBuffer(image.data(), sycl::range<1>(width * height));
    sycl::buffer<int> kernelBuffer(kernel.data(), sycl::range<1>(3 * 3));
    sycl::buffer<int> resultBuffer(result.data(), sycl::range<1>(width * height));

    // Submit kernel
    queue.submit([&](sycl::handler& cgh) {
        auto imageAcc  = imageBuffer.get_access<sycl::access::mode::read>(cgh);
        auto kernelAcc = kernelBuffer.get_access<sycl::access::mode::read>(cgh);
        auto resultAcc = resultBuffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class sobel_filter>(
            sycl::range<2>(width, height),
            [=](sycl::item<2> item) {
                int sum = 0;
                for (int kx = -1; kx <= 1; ++kx) {
                    for (int ky = -1; ky <= 1; ++ky) {
                        int x = item[0] + kx;
                        int y = item[1] + ky;

                        if (x >= 0 && x < width && y >= 0 && y < height) {
                            int imageIndex = x * width + y;
                            int kernelIndex = (kx + 1) * 3 + (ky + 1);
                            sum += imageAcc[imageIndex] * kernelAcc[kernelIndex];
                        }
                    }
                }
                resultAcc[item[0] * width + item[1]] = sum;
            });
    });

    queue.wait();

    // Access result
    auto hostResult = resultBuffer.get_access<sycl::access::mode::read>();

    // Display original image
    cout << "Image Matrix:\n";
    for (size_t i = 0; i < height; ++i) {
        cout << "\t";
        for (size_t j = 0; j < width; ++j)
            cout << image[i * width + j] << "\t";
        cout << "\n";
    }

    // Display result
    cout << "\nSobel Convolution Result:\n";
    for (size_t i = 0; i < height; ++i) {
        cout << "\t";
        for (size_t j = 0; j < width; ++j)
            cout << hostResult[i * width + j] << "\t";
        cout << "\n";
    }

    cout << "============================================\n";
    return 0;
}

