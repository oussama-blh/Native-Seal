#include "square.h"
#include <stdexcept>
#include <iostream>
#include <omp.h>

SquareLayer::SquareLayer(CKKSPyfhel &he) : he_(he) {
    // Ensure relinearization keys exist
    if (he_.get_relin_keys().data().empty()) {
        throw std::runtime_error("Relinearization keys not generated! Call generate_relin_keys() first.");
    }
    relin_keys_ = he_.get_relin_keys();
}

// Perform square operation on a single ciphertext in place
void SquareLayer::square_inplace(seal::Ciphertext &ct) {

    // Apply square operation
    he_.evaluator_->square(ct, ct);  // Modify the original ciphertext `ct` in place
    
    // Relinearize using pre-stored keys
    he_.evaluator_->relinearize_inplace(ct, relin_keys_);

    // Rescale only if necessary
    if (ct.is_ntt_form()) {
        he_.evaluator_->rescale_to_next_inplace(ct);
    }
}

// Square operation on a 1D vector (modifies input directly)
void SquareLayer::operator()(std::vector<seal::Ciphertext> &input) {
    for (auto &ct : input) {
        square_inplace(ct);  // Modify input directly
    }
}

// Square operation on a 4D tensor (modifies input directly)
void SquareLayer::operator()(std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input) {
    // Use index-based loops with collapse(4) to parallelize all four nested loops
    #pragma omp parallel for collapse(4)
    for (size_t i = 0; i < input.size(); i++) {
        for (size_t j = 0; j < input[i].size(); j++) {
            for (size_t k = 0; k < input[i][j].size(); k++) {
                for (size_t l = 0; l < input[i][j][k].size(); l++) {
                    square_inplace(input[i][j][k][l]);
                }
            }
        }
    }
}
