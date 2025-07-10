#include <paras/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

namespace sycl = paras::sycl;

int main(int argc, char* argv[]) {
    
    std::string input_file = argv[1];
    std::string output_file = argv[2];

    // Load grayscale image
    cv::Mat img_in = cv::imread(input_file, cv::IMREAD_GRAYSCALE);
    
    int width = img_in.cols;
    int height = img_in.rows;

    cv::Mat img_out(height, width, CV_8UC1);

    sycl::queue q;
    std::cout << "Running on: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    constexpr uint8_t threshold = 128;

    {
        sycl::buffer<uint8_t, 1> buf_in(img_in.data, sycl::range<1>(width * height));
        sycl::buffer<uint8_t, 1> buf_out(img_out.data, sycl::range<1>(width * height));

        q.submit([&](sycl::handler& h) {
            auto in = buf_in.get_access<sycl::access::mode::read>(h);
            auto out = buf_out.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<1>(width * height), [=](sycl::id<1> idx) {
                out[idx] = (in[idx] > threshold) ? 255 : 0;
            });
        });
        q.wait();
    }

    cv::imwrite(output_file, img_out);

    std::cout << "Output written to: " << output_file << "\n";

    return 0;
}

