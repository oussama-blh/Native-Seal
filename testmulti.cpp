#include <iostream>
#include <iomanip>  // For formatting output
#include "he/he.h"  // CKKS encryption header
#include "convolution/convolution.h"  // Convolution layer
#include "pooling/avgpooling.h"  // AvgPooling layer

// Function to decrypt and display final results
void displayFinalResult(CKKSPyfhel &he, const std::vector<std::vector<std::vector<seal::Ciphertext>>> &pooledOutput) {
    for (size_t f = 0; f < pooledOutput.size(); f++) {
        std::cout << "\nFinal Decrypted Output for Filter " << f + 1 << ":\n";
        int outHeight = static_cast<int>(pooledOutput[f].size());
        int outWidth = (outHeight > 0) ? static_cast<int>(pooledOutput[f][0].size()) : 0;

        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                double val = he.decrypt(pooledOutput[f][y][x]);
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
            }
            std::cout << std::endl;
        }
    }
}

int main() {
    // 1) Initialize CKKS and generate keys
    CKKSPyfhel he;
    he.generate_keys();
    he.generate_relin_keys();

    // 2) Define a single input image (static matrix for testing)
    std::vector<std::vector<std::vector<std::vector<double>>>> inputDouble(1);
    inputDouble[0].resize(1);
    inputDouble[0][0] = {
        {1.0, 2.0, 3.0, 4.0},
        {5.0, 6.0, 7.0, 8.0},
        {9.0, 10.0, 11.0, 12.0},
        {13.0, 14.0, 15.0, 16.0}
    };

    // 3) Define a convolution layer with multiple output channels
    std::vector<std::vector<std::vector<double>>> weights {
        {
            
            {1.0, 0.5, -1.0},
            {0.5, 2.0, 0.5},
            {-1.0, 0.5, 1.0}
        },
        {
            {-0.5, 1.0, 0.5},
            {1.5, -1.0, -0.5},
            {0.5, 1.0, -1.5}
        }
    };
    std::vector<double> bias { 0.1, -0.2 };  // One bias per filter

    Conv2d convLayer(
        he, weights, 
        std::make_pair(1, 1),  // Stride
        std::make_pair(1, 1),  // Padding
        bias
    );

    // 4) Encrypt the input image
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> inputEnc(1);
    inputEnc[0].resize(1);
    inputEnc[0][0] = he.encryptMatrix2D(inputDouble[0][0]); // Encrypt the input

    // 5) Apply the convolution layer (multiple output channels)
    auto convOutput = convLayer(inputEnc);

    std::cout << "Convolution Output Shape: [" << convOutput.size() << "][" 
              << convOutput[0].size() << "][" 
              << convOutput[0][0].size() << "][" 
              << convOutput[0][0][0].size() << "]\n";

    // 8) Decrypt and print the final convolution result
    auto &out2D = convOutput[0];
    int outHeight = static_cast<int>(out2D[0].size());
    int outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0][0].size()) : 0;

    std::cout << "\nFinal Decrypted Convolution Output:\n";
    for (size_t f = 0; f < out2D.size(); f++) {
        std::cout << "\nFilter " << f + 1 << ":\n";
        for (int y = 0; y < outHeight; y++) {
            for (int x = 0; x < outWidth; x++) {
                double val = he.decrypt(out2D[f][y][x]);
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << val << " ";
            }
            std::cout << std::endl;
        }
    }

    // 6) Define Avg Pooling Layer (2x2 kernel, stride=2, padding=0)
    AvgPoolLayer avgPool(he, {2, 2}, {2, 2}, {0, 0});

    // 7) Apply Avg Pooling on convolution output
    auto pooledOutput = avgPool(convOutput);

    // 8) Decrypt and display the final result
    displayFinalResult(he, pooledOutput[0]); // Display for first image

    return 0;
}
