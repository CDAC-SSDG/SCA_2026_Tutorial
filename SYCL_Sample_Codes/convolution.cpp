#include <sycl.hpp>
#include <iostream>
#include <vector>

using namespace paras;  

int main() {
    // Take width and height input from user (static for now)
    size_t width = 3, height = 3;

    // Initialize the image vector with sequential values
    std::vector<int> image(width * height);
    for (size_t i = 0; i < width * height; ++i) {
        image[i] = i + 1;
    }

    // Sobel kernel (horizontal edge detection)
    const std::vector<int> kernel = {
        1, 0, -1,
        2, 0, -2,
        1, 0, -1
    };

    // Result vector initialized to zero
    std::vector<int> result(width * height, 0);

    // Create a SYCL queue
    sycl::queue queue{sycl::default_selector{}};

    std::cout << "SYCL device: " 
              << queue.get_device().get_info<sycl::info::device::name>() 
              << std::endl;

    // Create buffers
    sycl::buffer<int> imageBuffer(image.data(), sycl::range<1>(width * height));
    sycl::buffer<int> kernelBuffer(kernel.data(), sycl::range<1>(3 * 3));
    sycl::buffer<int> resultBuffer(result.data(), sycl::range<1>(width * height));

    // Submit the kernel
    queue.submit([&](sycl::handler& cgh) {
        auto imageAccessor = imageBuffer.get_access<sycl::access::mode::read>(cgh);
        auto kernelAccessor = kernelBuffer.get_access<sycl::access::mode::read>(cgh);
        auto resultAccessor = resultBuffer.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for<class image_convolution>(
            sycl::range<2>({width, height}),
            [=](sycl::item<2> item) {
                int sum = 0;
                for (int kx = -1; kx <= 1; ++kx) {
                    for (int ky = -1; ky <= 1; ++ky) {
                        int imageX = item[0] + kx;
                        int imageY = item[1] + ky;
                        if (imageX >= 0 && imageX < width && imageY >= 0 && imageY < height) {
                            sum += imageAccessor[imageX * width + imageY] *
                                   kernelAccessor[(kx + 1) * 3 + (ky + 1)];
                        }
                    }
                }
                resultAccessor[item[0] * width + item[1]] = sum;
            });
    });

    // Wait for the queue to finish
    queue.wait();

    // Access and print the result
    auto resultBuffer_host = resultBuffer.get_access<sycl::access::mode::read>();

    std::cout << "Convolution Result:" << std::endl;
    for (size_t i = 0; i < height; ++i) {
        for (size_t j = 0; j < width; ++j) {
            std::cout << resultBuffer_host[i * width + j] << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}

