#include <iostream>
#include <vector>
#include <iomanip>  // For formatted output
#include "he/he.h"           // CKKS encryption
#include "convolution/convolution.h"  // Convolution implementation
#include "pooling/avgpooling.h"  // AvgPooling implementation
#include "flatten/flatten.h"  // FlattenLayer implementation
#include "linear/linear.h"    // LinearLayer implementation
#include "pooling/adaptiveAvgPooling.h"  // Include AdaptiveAvgPoolLayer
#include "functions/square.h"

int main() {
    // 1) Create an instance of CKKS & generate keys
    CKKSPyfhel he;
    he.generate_keys();
    he.generate_relin_keys();

    // 2) Define a **static** 4x4 matrix as an image
    std::vector<std::vector<std::vector<std::vector<double>>>> inputDouble(1);
    inputDouble[0].resize(1); // 1 channel
    inputDouble[0][0] = {
        { 1.0, 2.0, 3.0, 4.0 },
        { 5.0, 6.0, 7.0, 8.0 },
        { 9.0, 10.0, 11.0, 12.0 },
        { 13.0, 14.0, 15.0, 16.0 }
    };

    // 3) Define the first convolution layer (3x3 kernel, stride=1, padding=1)
    std::vector<std::vector<std::vector<double>>> weights1 {
        {
            
            { 1.0, 0.5, -1.0 },
            { 0.5, 2.0, 0.5 },
            { -1.0, 0.5, 1.0 }
            
        }
    };
    std::vector<double> bias1 { 0.1 };

    Conv2d convLayer1(
        he, weights1, 
        std::make_pair(1, 1),  // stride
        std::make_pair(1, 1),  // padding
        bias1
    );

    // 4) Encrypt the input image
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> inputEnc(1);
    inputEnc[0].resize(1);
    inputEnc[0][0] = he.encryptMatrix2D(inputDouble[0][0]); // Encrypt the static image

    // 5) Apply first convolution layer
    auto outputEnc1 = convLayer1(inputEnc);

    // 8) Decrypt and print the final convolution result
    auto &out2D = outputEnc1[0][0];
    int outHeight = static_cast<int>(out2D.size());
    int outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

    std::cout << "\n First Decrypted Convolution Output:\n";
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            double val = he.decrypt(out2D[y][x]);
            std::cout << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }

    
    // 6) Define the second convolution layer (2x2 kernel, stride=1, padding=0)
    std::vector<std::vector<std::vector<double>>> weights2 {
        {
            { 0.5, -0.5 },
            { 1.0,  0.5 }
        }
    };
    std::vector<double> bias2 { -0.2 };

    Conv2d convLayer2(
        he, weights2, 
        std::make_pair(1, 1),  // stride
        std::make_pair(0, 0),  // padding
        bias2
    );

    // 7) Apply second convolution layer on output of first layer
    auto outputEnc2 = convLayer2(outputEnc1);

    

    // 8) Decrypt and print the final convolution result
    out2D = outputEnc2[0][0];
    outHeight = static_cast<int>(out2D.size());
    outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

    std::cout << "\nFinal Decrypted Convolution Output:\n";
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            double val = he.decrypt(out2D[y][x]);
            std::cout << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }

    
    
    // 9) Apply **Adaptive Average Pooling** after second convolution
    AdaptiveAvgPoolLayer adaptivePool(he, {1, 1}); // Output size (1x1)
    auto adaptivepooledOutput = adaptivePool(outputEnc2);

    // 10) Decrypt and print the Adaptive Pooling result
    std::cout << "\nDecrypted Adaptive Pooling Output:\n";
    auto &pooled2D = adaptivepooledOutput[0][0]; // First image, first channel
    int pooledHeight = static_cast<int>(pooled2D.size());
    int pooledWidth  = (pooledHeight > 0) ? static_cast<int>(pooled2D[0].size()) : 0;

    for (int y = 0; y < pooledHeight; y++) {
        for (int x = 0; x < pooledWidth; x++) {
            double val = he.decrypt(pooled2D[y][x]);
            std::cout << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }
    
    
    // Create FlattenLayer
    FlattenLayer flatten;

    // Apply flattening
    auto flattened_output = flatten(outputEnc2);

    // Print shape
    std::cout << "Flattened Output Size: [" << flattened_output.size() << "]\n";

    
    std::cout << "\nFlattened Vector Output:\n";

    for (size_t img = 0; img < flattened_output.size(); img++) // Loop over images in batch
    {
        std::cout << "Image " << img << ": ";
        for (size_t i = 0; i < flattened_output[img].size(); i++) // Loop over flattened vector
        {
            double decrypted_value = he.decrypt(flattened_output[img][i]);  //  Decrypt each value
            std::cout << std::fixed << std::setprecision(4) << decrypted_value << " ";
            
            if ((i + 1) % 10 == 0) {  // Print in rows of 10 for readability
                std::cout << "\n";
            }
        }
        std::cout << "\n";
    }

    // 1️ Define Linear Layer (2 output features for testing)
    std::vector<std::vector<double>> linear_weights = {
        { 0.5, -0.3, 1.2, 0.7, -0.6, 0.9, 0.1, -0.2, 0.8 }  // 1 row: 9 input features -> 1 output feature
    };
    std::vector<double> linear_bias = { 0.5 };  // Bias for output feature

    LinearLayer linearLayer(he, linear_weights, linear_bias);

    // 2️ Prepare input: Convert flattened output to `std::vector<std::vector<seal::Ciphertext>>`
    std::vector<std::vector<seal::Ciphertext>> linear_input = { flattened_output };  // Batch size 1

    // 3️ Pass through Linear Layer
    auto linear_output = linearLayer(linear_input);

    // 4️ Decrypt and print result
    std::cout << "\nLinear Layer Output:\n";
    for (size_t i = 0; i < linear_output[0].size(); i++) {
        double decrypted_value = he.decrypt(linear_output[0][i]);
        std::cout << std::fixed << std::setprecision(4) << decrypted_value << " ";
    }
    std::cout << std::endl;


    // Define Avg Pooling Layer (2x2 kernel, stride=2, padding=0)
    AvgPoolLayer avgPool(he, {2, 2}, {2, 2}, {0, 0});

    // Apply Avg Pooling on second convolution output
    auto pooledOutput = avgPool(outputEnc2);

    out2D = pooledOutput[0][0];
    outHeight = static_cast<int>(out2D.size());
    outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

    std::cout << "\n Decrypted Pooling Output:\n";
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            double val = he.decrypt(out2D[y][x]);
            std::cout << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }

   


    //Apply SquareLayer after the first convolution
    auto squareLayer = SquareLayer(he);

    auto squaredEnc = squareLayer(outputEnc1);

    out2D = squaredEnc[0][0];
    outHeight = static_cast<int>(out2D.size());
    outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

    std::cout << "\nFirst Decrypted Square Output:\n";
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            double val = he.decrypt(out2D[y][x]);
            std::cout << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }

    //Apply SquareLayer after the second convolution
    auto squaredEnc2 = squareLayer(outputEnc2);

    out2D = squaredEnc2[0][0];
    outHeight = static_cast<int>(out2D.size());
    outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

    std::cout << "\nSecond Decrypted Square Output:\n";
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            double val = he.decrypt(out2D[y][x]);
            std::cout << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}
