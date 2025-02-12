#include "linear.h"
#include <stdexcept>
#include <iostream>
#include <iomanip> 

// Constructor: Encodes Weights and Bias
LinearLayer::LinearLayer(CKKSPyfhel &he, const std::vector<std::vector<double>> &weights, 
                         const std::vector<double> &bias)
    : he_(he)
{
    // Encode weights
    weights_.resize(weights.size());
    for (size_t i = 0; i < weights.size(); i++) {
        weights_[i].resize(weights[i].size());
        for (size_t j = 0; j < weights[i].size(); j++) {
            weights_[i][j] = he_.encode(weights[i][j]);
        }
    }

    // Encode bias if provided
    if (!bias.empty()) {
        bias_.resize(bias.size());
        for (size_t i = 0; i < bias.size(); i++) {
            bias_[i] = he_.encode(bias[i]);
        }
    }
}

// Forward pass: Encrypted matrix-vector multiplication
std::vector<std::vector<seal::Ciphertext>> 
LinearLayer::operator()(const std::vector<std::vector<seal::Ciphertext>> &input)
{
    size_t n_samples = input.size();     // Number of input samples (batch size)
    size_t in_features = input[0].size(); // Input vector size
    size_t out_features = weights_.size(); // Output vector size

    // Ensure input matches weight dimensions
    if (in_features != weights_[0].size()) {
        throw std::runtime_error("LinearLayer Error: Input size does not match weight dimensions.");
    }

    std::vector<std::vector<seal::Ciphertext>> result(n_samples, std::vector<seal::Ciphertext>(out_features));

    for (size_t img = 0; img < n_samples; img++) {
        for (size_t out_f = 0; out_f < out_features; out_f++) {
            // Initialize sum to zero ciphertext
            seal::Ciphertext sum_ct = he_.encrypt(0.0);

            for (size_t in_f = 0; in_f < in_features; in_f++) {
                seal::Ciphertext prod_ct;
                
                he_.evaluator_->mod_switch_to_inplace(weights_[out_f][in_f], input[img][in_f].parms_id());
                weights_[out_f][in_f].scale() = input[img][in_f].scale();
                he_.evaluator_->multiply_plain(input[img][in_f], weights_[out_f][in_f], prod_ct);

                // Rescale and switch modulus if necessary
                he_.evaluator_->rescale_to_next_inplace(prod_ct);
                he_.evaluator_->mod_switch_to_inplace(sum_ct, prod_ct.parms_id());
                sum_ct.scale() = prod_ct.scale();

                // Sum the weighted input
                he_.evaluator_->add_inplace(sum_ct, prod_ct);
            }

            // Add bias if provided
            if (!bias_.empty()) {
                he_.evaluator_->mod_switch_to_inplace(bias_[out_f], sum_ct.parms_id());
                bias_[out_f].scale() = sum_ct.scale();
                he_.evaluator_->add_plain_inplace(sum_ct, bias_[out_f]);
            }

            result[img][out_f] = std::move(sum_ct);
        }
    }

    return result;
}

// Getter function to retrieve encoded weights (for debugging)
std::vector<std::vector<seal::Plaintext>> LinearLayer::get_weights() const {
    return weights_;
}
