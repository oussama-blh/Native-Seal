#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "../he/he.h"
#include "../convolution/convolution.h"
#include "../pooling/avgpooling.h"
#include "../flatten/flatten.h"
#include "../linear/linear.h"
// #include "../pooling/adaptiveAvgPooling.h"
#include <vector>
#include <iostream>
#include <memory>
#include <stdexcept>

// Sequential model container to hold and process different layers
class Sequential {
public:
    explicit Sequential(CKKSPyfhel &he);

    // Add a layer to the sequential model
    void addLayer(std::shared_ptr<void> layer);

    // Forward propagation through convolutional, pooling, and flatten layers
    std::vector<std::vector<seal::Ciphertext>>  operator()(std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &x);

    // Forward propagation through linear layers
    std::vector<std::vector<seal::Ciphertext>> operator()(std::vector<std::vector<seal::Ciphertext>>  &x);

    // Retrieve the last feature map (output of last Conv2d layer)
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>* getFeatureMap();

    // Retrieve the last embedding vector (output of Flatten layer)
    // std::vector<std::vector<seal::Ciphertext>>* getEmbedding();

private:
    CKKSPyfhel &he_;
    std::vector<std::shared_ptr<void>> layers_;
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>* feature_map_;
    std::vector<std::vector<seal::Ciphertext>>* embedding_;
};

#endif // SEQUENTIAL_H
