#include "pooling/AdaptiveAvgPooling.h"
#include <stdexcept>
#include <iostream>

AdaptiveAvgPoolLayer::AdaptiveAvgPoolLayer(CKKSPyfhel &he, std::pair<int, int> output_size)
    : he_(he), output_size_(output_size) {}

// Apply Adaptive Average Pooling on batch of encrypted images
std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>
AdaptiveAvgPoolLayer::operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input) {
    
    size_t n_images = input.size();
    size_t n_channels = (n_images > 0) ? input[0].size() : 0;

    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> result(n_images);

    for (size_t img = 0; img < n_images; img++) {
        result[img].resize(n_channels);
        for (size_t ch = 0; ch < n_channels; ch++) {
            result[img][ch] = adaptive_avg(input[img][ch]);
        }
    }
    return result;
}

// Perform Adaptive Average Pooling on a Single Channel
std::vector<std::vector<seal::Ciphertext>>
AdaptiveAvgPoolLayer::adaptive_avg(const std::vector<std::vector<seal::Ciphertext>> &image) {
    size_t input_height = image.size();
    size_t input_width = (input_height > 0) ? image[0].size() : 0;
    size_t target_height = output_size_.first;
    size_t target_width = output_size_.second;

    if (target_height == 0 || target_width == 0) {
        throw std::invalid_argument("Adaptive pooling output size must be non-zero.");
    }

    // Compute kernel size and stride
    int kernel_height = input_height / target_height;
    int kernel_width = input_width / target_width;
    std::pair<int, int> kernel_size = { kernel_height, kernel_width };
    std::pair<int, int> stride = kernel_size;

    // Prepare encoded denominator (1 / kernel_size)
    double scale_factor = 1.0 / (kernel_size.first * kernel_size.second);
    seal::Plaintext denominator = he_.encode(scale_factor);

    // Initialize pooled result
    std::vector<std::vector<seal::Ciphertext>> pooled(target_height, std::vector<seal::Ciphertext>(target_width));

    for (size_t y = 0; y < target_height; y++) {
        for (size_t x = 0; x < target_width; x++) {
            // Aggregate values inside kernel
            seal::Ciphertext sum_ct;
            bool first = true;

            for (size_t ky = 0; ky < kernel_size.first; ky++) {
                for (size_t kx = 0; kx < kernel_size.second; kx++) {
                    size_t idx_y = y * stride.first + ky;
                    size_t idx_x = x * stride.second + kx;

                    if (idx_y < input_height && idx_x < input_width) {
                        if (first) {
                            sum_ct = image[idx_y][idx_x]; // Initialize sum
                            first = false;
                        } else {
                            he_.evaluator_->add_inplace(sum_ct, image[idx_y][idx_x]); // Sum pixels
                            
                        }
                    }
                }
            }

            // Compute average (sum * (1/kernel_size))
            he_.evaluator_->mod_switch_to_inplace(denominator, sum_ct.parms_id());
            denominator.scale() = sum_ct.scale();  

            he_.evaluator_->multiply_plain_inplace(sum_ct, denominator);
            he_.evaluator_->rescale_to_next_inplace(sum_ct);

            std::cout << "Done ! \n";
            pooled[y][x] = std::move(sum_ct);
        }
    }
    return pooled;
}
