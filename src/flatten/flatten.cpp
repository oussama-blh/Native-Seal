#include "flatten.h"
#include <stdexcept> // Exception handling
#include <iostream>  // Debugging

std::vector<std::vector<seal::Ciphertext>> 
FlattenLayer::operator()(const std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> &input)
{
    // Ensure input is not empty
    size_t n_images = input.size();
    if (n_images == 0) {
        throw std::runtime_error("FlattenLayer Error: Input is empty!");
    }
    
    size_t n_channels = input[0].size();
    size_t height = (n_channels > 0) ? input[0][0].size() : 0;
    size_t width = (height > 0) ? input[0][0][0].size() : 0;
    
    size_t flattened_size = n_channels * height * width; // Per-image flattened size

    // Initialize output vector
    std::vector<std::vector<seal::Ciphertext>> output(n_images); // Batch of flattened images

    // Flatten each image independently
    for (size_t img = 0; img < n_images; img++)
    {
        output[img].reserve(flattened_size); // Optimize memory allocation

        for (size_t c = 0; c < n_channels; c++)
        {
            for (size_t y = 0; y < height; y++)
            {
                for (size_t x = 0; x < width; x++)
                {
                    output[img].push_back(input[img][c][y][x]); // Append ciphertext
                }
            }
        }
    }

    return output;
}
