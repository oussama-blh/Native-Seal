#ifndef SQUARE_LAYER_H
#define SQUARE_LAYER_H

#include "../he/he.h"
#include <vector>
#include <seal/seal.h>

class SquareLayer {
public:
    explicit SquareLayer(CKKSPyfhel &he);

    // Applies the square function in-place on a 1D vector of encrypted ciphertexts
    void operator()(std::vector<seal::Ciphertext> &input);

    // Applies the square function in-place on a 4D tensor of encrypted ciphertexts
    void operator()(std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input);

private:
    CKKSPyfhel &he_;
    seal::RelinKeys relin_keys_;
    
    // Function to perform the square operation in-place
    void square_inplace(seal::Ciphertext &ct);
};

#endif // SQUARE_LAYER_H
