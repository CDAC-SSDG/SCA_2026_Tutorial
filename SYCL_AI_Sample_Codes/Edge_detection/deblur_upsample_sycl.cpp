#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "stb_image.h"
#include "stb_image_write.h"

#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace paras;
int main() {
    int width, height, channels;

    // Load grayscale image
    unsigned char* input = stbi_load("input1.jpg", &width, &height, &channels, 1);
    if (!input) {
        std::cerr << "Failed to load input image\n";
        return 1;
    }

    std::vector<unsigned char> output(width * height);
    sycl::queue q;

    {
        sycl::buffer<unsigned char, 2> in_buf(input, sycl::range<2>(height, width));
        sycl::buffer<unsigned char, 2> out_buf(output.data(), sycl::range<2>(height, width));

        const int Gx[3][3] = {
            {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1}
        };

        const int Gy[3][3] = {
            {-1, -2, -1},
            { 0,  0,  0},
            { 1,  2,  1}
        };

        q.submit([&](sycl::handler& h) {
            auto in = in_buf.get_access<sycl::access::mode::read>(h);
            auto out = out_buf.get_access<sycl::access::mode::write>(h);

            h.parallel_for(sycl::range<2>(height, width), [=](sycl::id<2> idx) {
                int y = idx[0];
                int x = idx[1];
                int gx = 0, gy = 0;

                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int ny = sycl::clamp(y + ky, 0, height - 1);
                        int nx = sycl::clamp(x + kx, 0, width - 1);
                        gx += in[ny][nx] * Gx[ky + 1][kx + 1];
                        gy += in[ny][nx] * Gy[ky + 1][kx + 1];
                    }
                }

                int mag = sycl::clamp((int)std::sqrt(gx * gx + gy * gy), 0, 255);
                out[y][x] = static_cast<unsigned char>(mag);
            });
        });
        q.wait();
    }

    stbi_write_png("edge_output1.jpg", width, height, 1, output.data(), width);
    stbi_image_free(input);
    std::cout << "Saved edge_output.png\n";

    return 0;
}

