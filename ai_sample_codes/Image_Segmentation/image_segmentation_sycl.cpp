#include <paras/sycl.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <random>

namespace sycl = paras::sycl;

struct Centroid {
    float r, g, b;
};

int main(int argc, char* argv[]) {
   

    std::string input_file = argv[1];
    std::string output_file = argv[2];

    cv::Mat img = cv::imread(input_file);
  
    int width = img.cols;
    int height = img.rows;
    int n_pixels = width * height;

    const int K = 3; // number of clusters
    std::vector<Centroid> centroids(K);

    // Randomly initialize centroids
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n_pixels - 1);
    for (int k = 0; k < K; ++k) {
        cv::Vec3b color = img.at<cv::Vec3b>(dist(rng) / width, dist(rng) % width);
        centroids[k] = {static_cast<float>(color[2]), static_cast<float>(color[1]), static_cast<float>(color[0])};
    }

    std::vector<int> labels(n_pixels, 0);
    std::vector<Centroid> new_centroids(K);

    sycl::queue q;

    std::cout << "Running on: " 
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    bool converged = false;
    int iterations = 0;

    while (!converged && iterations < 20) {
        iterations++;

        {
            sycl::buffer<cv::Vec3b, 1> buf_img(img.ptr<cv::Vec3b>(), sycl::range<1>(n_pixels));
            sycl::buffer<int, 1> buf_labels(labels.data(), sycl::range<1>(n_pixels));
            sycl::buffer<Centroid, 1> buf_centroids(centroids.data(), sycl::range<1>(K));

            q.submit([&](sycl::handler& h) {
                auto img_acc = buf_img.get_access<sycl::access::mode::read>(h);
                auto label_acc = buf_labels.get_access<sycl::access::mode::write>(h);
                auto cent_acc = buf_centroids.get_access<sycl::access::mode::read>(h);

                h.parallel_for(sycl::range<1>(n_pixels), [=](sycl::id<1> idx) {
                    cv::Vec3b color = img_acc[idx];
                    float min_dist = FLT_MAX;
                    int best_k = 0;

                    for (int k = 0; k < K; ++k) {
                        float dr = color[2] - cent_acc[k].r;
                        float dg = color[1] - cent_acc[k].g;
                        float db = color[0] - cent_acc[k].b;
                        float dist = dr * dr + dg * dg + db * db;
                        if (dist < min_dist) {
                            min_dist = dist;
                            best_k = k;
                        }
                    }
                    label_acc[idx] = best_k;
                });
            });
            q.wait();
        }

        // Recompute centroids 
        std::vector<int> counts(K, 0);
        new_centroids.assign(K, {0.0f, 0.0f, 0.0f});

        for (int i = 0; i < n_pixels; ++i) {
            int k = labels[i];
            cv::Vec3b color = img.at<cv::Vec3b>(i / width, i % width);
            new_centroids[k].r += color[2];
            new_centroids[k].g += color[1];
            new_centroids[k].b += color[0];
            counts[k]++;
        }

        converged = true;
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                new_centroids[k].r /= counts[k];
                new_centroids[k].g /= counts[k];
                new_centroids[k].b /= counts[k];
            }
            if (std::abs(new_centroids[k].r - centroids[k].r) > 1.0f ||
                std::abs(new_centroids[k].g - centroids[k].g) > 1.0f ||
                std::abs(new_centroids[k].b - centroids[k].b) > 1.0f) {
                converged = false;
            }
        }

        centroids = new_centroids;
    }

    // Assign final cluster colors
    for (int i = 0; i < n_pixels; ++i) {
        int k = labels[i];
        img.at<cv::Vec3b>(i / width, i % width) = cv::Vec3b(
            static_cast<uint8_t>(centroids[k].b),
            static_cast<uint8_t>(centroids[k].g),
            static_cast<uint8_t>(centroids[k].r)
        );
    }

    cv::imwrite(output_file, img);
    std::cout << "Segmented image written to: " << output_file << "\n";

    return 0;
}
