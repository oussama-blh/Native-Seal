#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include <vector>
#include "../he/he.h"  // Include your CKKS encryption header

class LinearLayer {
public:
    // Constructor: Takes HE reference, weights (2D vector), and optional bias
    LinearLayer(CKKSPyfhel &he, const std::vector<std::vector<double>> &weights, 
                const std::vector<double> &bias = {});

    // Forward pass
    std::vector<std::vector<seal::Ciphertext>> operator()(const std::vector<std::vector<seal::Ciphertext>> &input);

    // Getter for weights (for debugging)
    std::vector<std::vector<seal::Plaintext>> get_weights() const;

private:
    CKKSPyfhel &he_;  // Homomorphic Encryption object
    std::vector<std::vector<seal::Plaintext>> weights_; // Encoded Weights
    std::vector<seal::Plaintext> bias_; // Encoded Bias
};

#endif
