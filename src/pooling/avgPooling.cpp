#include "avgpooling.h"
#include "convolution/convolution.h"  // For apply_padding
#include <iostream>

// Constructor
AvgPoolLayer::AvgPoolLayer(CKKSPyfhel &he, std::pair<int, int> kernel_size, std::pair<int, int> stride, std::pair<int, int> padding)
    : he_(he), kernel_size_(kernel_size), stride_(stride), padding_(padding) {}

// Forward pass
std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> AvgPoolLayer::operator()(
    const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input)
{
    auto padded_input = apply_padding(input, padding_, he_);

    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> result;
    result.resize(padded_input.size());  // Number of images

    for (size_t img = 0; img < padded_input.size(); img++) {
        result[img].resize(padded_input[img].size());  // Number of layers
        for (size_t layer = 0; layer < padded_input[img].size(); layer++) {
            result[img][layer] = avg(he_, padded_input[img][layer], kernel_size_, stride_);
        }
    }

    return result;
}

// Avg Pooling Function for a 2D Image
std::vector<std::vector<seal::Ciphertext>> AvgPoolLayer::avg(
    CKKSPyfhel &he, 
    const std::vector<std::vector<seal::Ciphertext>> &image, 
    std::pair<int, int> kernel_size, 
    std::pair<int, int> stride)
{
    int y_s = stride.first;
    int x_s = stride.second;

    int y_k = kernel_size.first;
    int x_k = kernel_size.second;

    int y_d = image.size();
    int x_d = (y_d > 0) ? image[0].size() : 0;

    int y_o = ((y_d - y_k) / y_s) + 1;
    int x_o = ((x_d - x_k) / x_s) + 1;

    // Create plaintext for division factor
    seal::Plaintext denominator = he.encode(1.0 / (x_k * y_k));

    std::vector<std::vector<seal::Ciphertext>> result(y_o, std::vector<seal::Ciphertext>(x_o));

    for (int y = 0; y < y_o; y++) {
        for (int x = 0; x < x_o; x++) {
            // Extract sub-matrix and compute sum
            seal::Ciphertext sum_ct = he.encrypt(0.0);  // Initialize to zero
            for (int fy = 0; fy < y_k; fy++) {
                for (int fx = 0; fx < x_k; fx++) {
                    int row = y * y_s + fy;
                    int col = x * x_s + fx;

                    // Accumulate sum
                    he.evaluator_->mod_switch_to_inplace(sum_ct, image[row][col].parms_id());
                    sum_ct.scale() = image[row][col].scale();  // important for CKKS
                    he.evaluator_->add_inplace(sum_ct, image[row][col]);
                }
            }

            // Multiply by denominator (scaling)
            he.evaluator_->mod_switch_to_inplace(denominator, sum_ct.parms_id());
            denominator.scale() = sum_ct.scale();  // important for CKKS
            he.evaluator_->multiply_plain_inplace(sum_ct, denominator);
            he.evaluator_->rescale_to_next_inplace(sum_ct);

            // Store result
            result[y][x] = sum_ct;
        }
    }

    return result;
}
