#include "sequential.h"
#include "../pooling/avgPooling.h"  
#include "../functions/square.h"  

// Constructor
Sequential::Sequential(CKKSPyfhel &he) : he_(he), feature_map_(nullptr), embedding_(nullptr) {}

// Add a layer to the sequential model
void Sequential::addLayer(std::shared_ptr<void> layer) {
    layers_.push_back(layer);
}

// Forward propagation through convolutional, pooling, flatten, and square layers
std::vector<std::vector<seal::Ciphertext>>  
Sequential::operator()(std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &x) {
    for (auto &layer : layers_) {
        if (auto conv = std::dynamic_pointer_cast<Conv2d>(layer)) {
            x = (*conv)(x);
            feature_map_ = &x; // Store the feature map after the last Conv2d layer
        } else if (auto pool = std::dynamic_pointer_cast<AvgPoolLayer>(layer)) {
            x = (*pool)(x);
        // } else if (auto adaptive_pool = std::dynamic_pointer_cast<AvgPoolLayer>(layer)) {
        //     x = (*adaptive_pool)(x);
        } else if (auto square = std::dynamic_pointer_cast<SquareLayer>(layer)) {
            x = (*square)(x); // Apply element-wise squaring
        } else if (auto flatten = std::dynamic_pointer_cast<FlattenLayer>(layer)) {
            
            auto flat_output = (*flatten)(x);
            embedding_ = &flat_output;
            return flat_output;
        }
    }
    throw std::runtime_error("Sequential Error: Model did not produce a valid flattened output.");
}

// Forward propagation for linear layers (takes a vector instead of a 4D tensor)
std::vector<std::vector<seal::Ciphertext>> Sequential::operator()(std::vector<std::vector<seal::Ciphertext>> &x) {
    for (auto &layer : layers_) {
        if (auto linear = std::dynamic_pointer_cast<LinearLayer>(layer)) {
            return (*linear)(x);
        }
    }
    throw std::runtime_error("Sequential Error: No Linear Layer found to process input.");
}

// Get the last feature map (output of the last Conv2d layer)
std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>>* Sequential::getFeatureMap() {
    return feature_map_;
}

// // Get the last embedding vector (output of the Flatten layer)
// std::vector<std::vector<seal::Ciphertext>>* Sequential::getEmbedding() {
//     return embedding_;
// }
