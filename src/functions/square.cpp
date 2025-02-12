#include "square.h"
#include <stdexcept>
#include <iostream>

SquareLayer::SquareLayer(CKKSPyfhel &he) : he_(he) {
    // Ensure relinearization keys exist
    if (he_.get_relin_keys().data().empty()) {
        throw std::runtime_error("Relinearization keys not generated! Call generate_relin_keys() first.");
    }
    relin_keys_ = he_.get_relin_keys();
}

// Perform square operation on a single ciphertext
seal::Ciphertext SquareLayer::square(const seal::Ciphertext &ct) {
    seal::Ciphertext squared_ct;

    // Ensure deep copy by explicitly creating a new ciphertext
    seal::Ciphertext temp_ct;
    temp_ct = ct;  

    // Apply square operation
    he_.evaluator_->square(temp_ct, squared_ct);
    
    // Relinearize using pre-stored keys
    he_.evaluator_->relinearize_inplace(squared_ct, relin_keys_);

    // Rescale only if necessary
    if (squared_ct.is_ntt_form()) {
        he_.evaluator_->rescale_to_next_inplace(squared_ct);
    }

    return squared_ct;  // Return a fresh ciphertext
}

// Square operation on a 1D vector (DOES NOT MODIFY INPUT)
std::vector<seal::Ciphertext> SquareLayer::operator()(const std::vector<seal::Ciphertext> &input) {
    std::vector<seal::Ciphertext> output;
    output.reserve(input.size());

    for (const auto &ct : input) {
        output.emplace_back(square(ct));  // Use emplace_back for efficiency
    }

    return output;  // Return a new vector
}

// Square operation on a 4D tensor (Ensuring Full Deep Copy)
std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> 
SquareLayer::operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input) 
{
    // Create a NEW 4D tensor
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> output;

    output.resize(input.size());

    for (size_t i = 0; i < input.size(); i++) {
        output[i].resize(input[i].size());
        for (size_t j = 0; j < input[i].size(); j++) {
            output[i][j].resize(input[i][j].size());
            for (size_t k = 0; k < input[i][j].size(); k++) {
                output[i][j][k].resize(input[i][j][k].size());

                for (size_t l = 0; l < input[i][j][k].size(); l++) {
                    // Create a deep copy before squaring
                    seal::Ciphertext temp_ct;
                    temp_ct = input[i][j][k][l];  // Explicitly copy input value
                    
                    // Store squared value in a new object
                    output[i][j][k][l] = square(temp_ct);
                }
            }
        }
    }

    return output;  // Return a deep-copied 4D tensor
}
