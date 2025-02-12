#ifndef FLATTEN_H
#define FLATTEN_H

#include "../he/he.h"  // Ensure correct path
#include <vector>
#include <seal/seal.h>

class FlattenLayer {
public:
    FlattenLayer() = default;

    // Function to flatten the encrypted tensor for each image independently
    std::vector<std::vector<seal::Ciphertext>> 
    operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input);
};

#endif // FLATTEN_H
