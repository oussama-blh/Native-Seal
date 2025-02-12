#ifndef SQUARE_LAYER_H
#define SQUARE_LAYER_H

#include "../he/he.h"
#include <vector>
#include <seal/seal.h>

class SquareLayer {
public:
    explicit SquareLayer(CKKSPyfhel &he);

    // Applies the square function on encrypted tensors
    std::vector<seal::Ciphertext> operator()(const std::vector<seal::Ciphertext> &input);

    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> 
    operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input);

private:
    CKKSPyfhel &he_;
    seal::RelinKeys relin_keys_;
    // Function to perform element-wise square
    seal::Ciphertext square(const seal::Ciphertext &ct);
};

#endif // SQUARE_LAYER_H
