#include "convolution.h"
#include <stdexcept>   // for exceptions
#include <iostream>    // for debug prints (optional)
#include <chrono>
#include <omp.h>

// Helper: multiply ciphertext by plaintext, returning a new ciphertext.
// Multiply a batch of ciphertexts by a batch of plaintexts in parallel
static std::vector<seal::Ciphertext> multiply_ciphertext_plain_batch(
    CKKSPyfhel &he,
    const std::vector<seal::Ciphertext> &ct_vec,
    const std::vector<seal::Plaintext> &pt_vec)
{
    if (ct_vec.size() != pt_vec.size())
        throw std::invalid_argument("Mismatched sizes for ciphertext and plaintext vectors.");

    std::vector<seal::Ciphertext> result(ct_vec.size());

    // Process in parallel
    #pragma omp parallel for
    for (int i = 0; i < ct_vec.size(); i++)
    {
        seal::Plaintext pt_aligned = pt_vec[i];
        he.evaluator_->mod_switch_to_inplace(pt_aligned, ct_vec[i].parms_id());
        pt_aligned.scale() = ct_vec[i].scale();
        he.evaluator_->multiply_plain(ct_vec[i], pt_aligned, result[i]);
        he.evaluator_->rescale_to_next_inplace(result[i]);
    }

    return result;
}

/*************************************************************
 * Conv2d Implementation
 *************************************************************/
Conv2d::Conv2d(
    CKKSPyfhel &he,
    const std::vector<std::vector<std::vector<std::vector<double>>>> &weights,
    std::pair<int,int> stride,
    std::pair<int,int> padding,
    const std::vector<double> &bias
)
  : he_(he), stride_(stride), padding_(padding)
{
    // Encode the 4D weights as Plaintext.
    // Expected shape: [n_filters][n_input_channels][kernel_height][kernel_width]

    // duration measurement
    auto start = std::chrono::high_resolution_clock::now();

    // Perform resizing before parallel execution
    weights_.resize(weights.size());
    #pragma omp parallel for
    for (size_t f = 0; f < weights.size(); f++) {
        weights_[f].resize(weights[f].size());
        for (size_t in_c = 0; in_c < weights[f].size(); in_c++) {
            weights_[f][in_c].resize(weights[f][in_c].size());
            for (size_t y = 0; y < weights[f][in_c].size(); y++) {
                weights_[f][in_c][y].resize(weights[f][in_c][y].size());
            }
        }
    }

    //the parallel loop to encode the weights
    #pragma omp parallel for collapse(4)
    for (size_t f = 0; f < weights.size(); f++) {
        for (size_t in_c = 0; in_c < weights[f].size(); in_c++) {
            for (size_t y = 0; y < weights[f][in_c].size(); y++) {
                for (size_t x = 0; x < weights[f][in_c][y].size(); x++) {
                    double w = weights[f][in_c][y][x];
                    weights_[f][in_c][y][x] = he_.encode(w);
                }
            }
        }
    }

    // Encode the bias if provided (assume bias.size() == n_filters)
    if (!bias.empty())
    {
        bias_.resize(bias.size());
        #pragma omp parallel for
        for (size_t i = 0; i < bias.size(); i++)
        {
            bias_[i] = he_.encode(bias[i]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    // For milliseconds:
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken for weights encoding: " << duration_ms.count() << " milliseconds" << std::endl;

}

std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>
Conv2d::operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input)
{
    // input shape = [n_images, n_input_channels, height, width]
    // weights_ shape = [n_filters][n_input_channels][kernel_height][kernel_width]

    auto padded_input = apply_padding(input, padding_, he_);
    size_t n_images = padded_input.size();
    size_t n_input_channels = (n_images > 0) ? padded_input[0].size() : 0;
    size_t n_filters = weights_.size(); // Number of output channels

    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> result;
    result.resize(n_images); // Number of images

    for (size_t img = 0; img < n_images; img++) { // Loop over images
        result[img].resize(n_filters); // Number of output channels

        // For each output channel f
        for (size_t f = 0; f < n_filters; f++) {
            std::vector<std::vector<seal::Ciphertext>> filter_sum_2d;

            // For each input channel in_c, apply the corresponding filter weights_[f][in_c]
            for (size_t in_c = 0; in_c < n_input_channels; in_c++) {
                // Use the weight corresponding to output channel f and input channel in_c.
                // Note: weights_[f][in_c] is a 2D vector of Plaintext.
                auto conv_layer = convolute2d(padded_input[img][in_c], weights_[f][in_c], stride_, he_); 
                if (in_c == 0) {
                    filter_sum_2d = conv_layer; 
                } else {
                    // Ensure dimensions match before summing.
                    if (filter_sum_2d.size() != conv_layer.size() || 
                        (filter_sum_2d.size() > 0 && filter_sum_2d[0].size() != conv_layer[0].size())) {
                        throw std::runtime_error("Mismatch in output size while summing channels.");
                    }
                    // Sum contributions from each input channel.
                    for (size_t yy = 0; yy < filter_sum_2d.size(); yy++) {
                        for (size_t xx = 0; xx < filter_sum_2d[yy].size(); xx++) {
                            he_.evaluator_->add_inplace(filter_sum_2d[yy][xx], conv_layer[yy][xx]); 
                        }
                    }
                }

                // conv_layer.clear();
                // conv_layer.shrink_to_fit();
            }

            // Add bias for this output channel if provided.
            if (!bias_.empty()) {
                auto &bias_pt = bias_[f];
            
                // Parallelize the nested loops
                #pragma omp parallel for collapse(2)
                for (size_t i = 0; i < filter_sum_2d.size(); ++i) {
                    for (size_t j = 0; j < filter_sum_2d[i].size(); ++j) {
                        auto &ciph = filter_sum_2d[i][j];
            
                        // OpenMP threads may cause race conditions if bias_pt is modified in place
                        // Ensure thread safety by working on a thread-local copy
                        auto bias_pt_local = bias_pt;
            
                        he_.evaluator_->mod_switch_to_inplace(bias_pt_local, ciph.parms_id());
                        bias_pt_local.scale() = ciph.scale();
                        he_.evaluator_->add_plain_inplace(ciph, bias_pt_local);
                    }
                }
            }

            std::cout << "Done \n" ;
            // Store the result for this output channel.
            result[img][f] = filter_sum_2d;

            // Free memory
            // for (auto& row : filter_sum_2d) {
            //     row.clear();  // Clear each inner vector
            //     row.shrink_to_fit();  // Free inner memory
            // }
            // filter_sum_2d.clear();  // Clear the outer vector
            // filter_sum_2d.shrink_to_fit();  // Free outer memory
        }
    }


    // padded_input.clear();
    // padded_input.shrink_to_fit();
    return result;
}

/*************************************************************
 * apply_padding
 *************************************************************/
std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>
apply_padding(
    const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input,
    std::pair<int,int> padding,
    CKKSPyfhel &he
)
{
    int y_pad = padding.first;
    int x_pad = padding.second;

    if (y_pad == 0 && x_pad == 0)
    {
        return input;
    }

    // Create a ciphertext that encrypts zero.
    seal::Ciphertext zero_ct = he.encrypt(0.0);

    auto output = input; // Copy input to output.
    for (size_t img = 0; img < output.size(); img++)
    {
        for (size_t l = 0; l < output[img].size(); l++)
        {
            size_t old_y = output[img][l].size();
            size_t old_x = (old_y > 0) ? output[img][l][0].size() : 0;
            size_t new_y = old_y + 2 * y_pad;
            size_t new_x = old_x + 2 * x_pad;

            std::vector<std::vector<seal::Ciphertext>> padded(
                new_y, std::vector<seal::Ciphertext>(new_x, zero_ct)
            );

            for (size_t yy = 0; yy < old_y; yy++)
            {
                for (size_t xx = 0; xx < old_x; xx++)
                {
                    padded[yy + y_pad][xx + x_pad] = output[img][l][yy][xx];
                }
            }

            output[img][l] = std::move(padded);
        }
    }

    return output;
}

/*************************************************************
 * convolute2d
 *************************************************************/
std::vector<std::vector<seal::Ciphertext>> convolute2d(
    const std::vector<std::vector<seal::Ciphertext>> &image,
    const std::vector<std::vector<seal::Plaintext>> &filter_matrix,
    std::pair<int, int> stride,
    CKKSPyfhel &he)
{
    int y_d = static_cast<int>(image.size());
    int x_d = (y_d > 0) ? static_cast<int>(image[0].size()) : 0;
    int y_f = static_cast<int>(filter_matrix.size());
    int x_f = (y_f > 0) ? static_cast<int>(filter_matrix[0].size()) : 0;

    if (y_f == 0 || x_f == 0)
        throw std::runtime_error("Kernel size is zero, cannot apply convolution.");
    if (stride.first <= 0 || stride.second <= 0)
        throw std::runtime_error("Stride must be positive.");
    if (y_d < y_f || x_d < x_f)
        throw std::runtime_error("Filter size is larger than input size.");

    int y_out = ((y_d - y_f) / stride.first) + 1;
    int x_out = ((x_d - x_f) / stride.second) + 1;

    if (y_out <= 0 || x_out <= 0)
        throw std::runtime_error("Output size is zero or negative. Check stride and padding.");

    std::vector<std::vector<seal::Ciphertext>> result(y_out, std::vector<seal::Ciphertext>(x_out));

    seal::Ciphertext zero_ct = he.encrypt(0.0);

    // Process the image in parallel for each output position
    #pragma omp parallel for collapse(2)
    for (int oy = 0; oy < y_out; oy++)
    {
        for (int ox = 0; ox < x_out; ox++)
        {
            int sub_y = oy * stride.first;
            int sub_x = ox * stride.second;

            std::vector<seal::Ciphertext> image_patch;
            std::vector<seal::Plaintext> filter_patch;
            image_patch.reserve(y_f * x_f);
            filter_patch.reserve(y_f * x_f);

            // Extract patch and filter values
            for (int fy = 0; fy < y_f; fy++)
            {
                for (int fx = 0; fx < x_f; fx++)
                {
                    image_patch.push_back(image[sub_y + fy][sub_x + fx]);
                    filter_patch.push_back(filter_matrix[fy][fx]);
                }
            }

            // Perform batch multiplication
            std::vector<seal::Ciphertext> products = multiply_ciphertext_plain_batch(he, image_patch, filter_patch);

            // Accumulate result
            seal::Ciphertext accum_ct = zero_ct;
            for (const auto &prod : products)
            {
                he.evaluator_->mod_switch_to_inplace(accum_ct, prod.parms_id());
                accum_ct.scale() = prod.scale();
                he.evaluator_->add_inplace(accum_ct, prod);
            }

            result[oy][ox] = std::move(accum_ct);

            // Explicitly free memory before next iteration
            // image_patch.clear();
            // image_patch.shrink_to_fit();
            // filter_patch.clear();
            // filter_patch.shrink_to_fit();
            // products.clear();         
            // products.shrink_to_fit(); 

        }
    }

    return result;
}
