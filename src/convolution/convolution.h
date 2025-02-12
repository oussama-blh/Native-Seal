#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <vector>
#include <utility>   // for std::pair
#include "he/he.h" // Your CKKSPyfhel class

/**
 * Conv2d class simulates a 2D convolution layer with homomorphic encryption.
 * - Weights and bias are stored as Plaintext arrays (encoded via CKKS).
 * - Inputs are ciphertext arrays.
 */
class Conv2d {
public:
    /**
     * @brief Constructor
     * @param he         Reference to your CKKSPyfhel (to encode weights, perform multiplications, etc.)
     * @param weights    4D raw double array [n_filters, n_input_channels, filter_height, filter_width]
     * @param stride     (y_stride, x_stride)
     * @param padding    (y_pad, x_pad)
     * @param bias       (optional) 1D array of double to encode as plaintext, length = n_filters
     */
    Conv2d(
        CKKSPyfhel &he,
        const std::vector<std::vector<std::vector<std::vector<double>>>> &weights,
        std::pair<int,int> stride = {1, 1},
        std::pair<int,int> padding = {0, 0},
        const std::vector<double> &bias = {}
    );

    /**
     * @brief Perform convolution on a batch of encrypted images.
     * @param input A 4D array of Ciphertext: [n_images, n_input_channels, height, width]
     * @return A 4D array of Ciphertext: [n_images, n_filters, out_height, out_width]
     */
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>
    operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input);

private:
    // Reference to the homomorphic encryption object
    CKKSPyfhel &he_;
    
    // 4D array of Plaintexts for weights.
    // [n_filters][n_input_channels][filter_height][filter_width]
    std::vector<std::vector<std::vector<std::vector<seal::Plaintext>>>> weights_;

    // 1D array of Plaintexts for bias, with length = n_filters.
    // If empty, no bias is used.
    std::vector<seal::Plaintext> bias_;

    // (y_stride, x_stride) and (y_padding, x_padding)
    std::pair<int,int> stride_;
    std::pair<int,int> padding_;
};

/**
 * @brief Zero-pad a batch of images.
 * @param input A 4D array: [n_images, n_channels, y, x]
 * @param padding (y_pad, x_pad)
 * @param he  Used to create a ciphertext that decrypts to zero.
 * @return The padded 4D array.
 */
std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>
apply_padding(
    const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input,
    std::pair<int,int> padding,
    CKKSPyfhel &he
);

/**
 * @brief 2D convolution between a ciphertext image and a plaintext filter.
 * @param image 2D ciphertext array [height, width]
 * @param filter_matrix 2D plaintext array [filter_height, filter_width]
 * @param stride (y_stride, x_stride)
 * @param he Used for HE operations (multiplyPlain, sum, etc.)
 * @return 2D ciphertext array [out_height, out_width]
 */
std::vector<std::vector<seal::Ciphertext>>
convolute2d(
    const std::vector<std::vector<seal::Ciphertext>> &image,
    const std::vector<std::vector<seal::Plaintext>> &filter_matrix,
    std::pair<int,int> stride,
    CKKSPyfhel &he
);

#endif // CONVOLUTION_H
