#ifndef AVGPOOLING_H
#define AVGPOOLING_H

#include "he/he.h"  // CKKS encryption header
#include <vector>
#include <seal/seal.h>

class AvgPoolLayer {
public:
    CKKSPyfhel &he_;
    std::pair<int, int> kernel_size_;
    std::pair<int, int> stride_;
    std::pair<int, int> padding_;

    // Constructor
    AvgPoolLayer(CKKSPyfhel &he, std::pair<int, int> kernel_size, std::pair<int, int> stride, std::pair<int, int> padding);

    // Forward pass
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> operator()(
        const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input);
    
private:
    // Perform average pooling on a 2D encrypted image
    std::vector<std::vector<seal::Ciphertext>> avg(
        CKKSPyfhel &he, 
        const std::vector<std::vector<seal::Ciphertext>> &image, 
        std::pair<int, int> kernel_size, 
        std::pair<int, int> stride);
};

#endif // AVGPOOLING_H
