#include <iostream>
#include <torch/torch.h>
#include "he/he.h"           // Your CKKS encryption header
#include "convolution/convolution.h"  // Your Conv2d implementation


// Convert a single MNIST image tensor to a 4D std::vector<double>
std::vector<std::vector<std::vector<std::vector<double>>>> tensorTo4DVector(torch::Tensor tensor) {
    tensor = tensor.squeeze().to(torch::kDouble); // Convert [1, 28, 28] -> [28, 28]

    int height = tensor.size(0);
    int width = tensor.size(1);

    std::vector<std::vector<std::vector<std::vector<double>>>> image(1, std::vector<std::vector<std::vector<double>>>(1, std::vector<std::vector<double>>(height, std::vector<double>(width))));

    for (int r = 0; r < height; r++) {
        for (int c = 0; c < width; c++) {
            image[0][0][r][c] = tensor[r][c].item<double>();
        }
    }
    return image;
}


int main() {
    // 1) Create an instance of CKKS & generate keys
    CKKSPyfhel he;
    he.generate_keys();
    he.generate_relin_keys();

    // 2) Load a real MNIST image
    std::string dataset_path = "C:/Khbich/PFE/Implementations/NativeSEAL/data/MNIST/raw";
    auto train_dataset = torch::data::datasets::MNIST(dataset_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))  // Normalize
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(1));

    torch::Tensor image_tensor;
    for (auto& batch : *train_loader) {
        image_tensor = batch.data[0]; // Get first image
        break;
    }

    // Convert to 4D vector
    auto inputDouble = tensorTo4DVector(image_tensor);

    // 3) Define the first convolution layer (3x3 kernel, stride=1, padding=1)
    std::vector<std::vector<std::vector<std::vector<double>>>> weights1 {
        {
            {
                { 1.0, 0.5, -1.0 },
                { 0.5, 2.0, 0.5 },
                { -1.0, 0.5, 1.0 }
            }
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
    inputEnc[0][0] = he.encryptMatrix2D(inputDouble[0][0]); // Encrypt the MNIST image

    // 5) Apply first convolution layer
    auto outputEnc1 = convLayer1(inputEnc);

    std::cout << "First Convolution Output Shape: [" << outputEnc1.size() << "][" 
          << outputEnc1[0].size() << "][" 
          << outputEnc1[0][0].size() << "][" 
          << outputEnc1[0][0][0].size() << "]\n";

    // 6) Define the second convolution layer (2x2 kernel, stride=1, padding=0)
    std::vector<std::vector<std::vector<std::vector<double>>>> weights2 {
        {
            {
                { 0.5, -0.5 },
                { 1.0,  0.5 }
            }
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
    auto &out2D = outputEnc2[0][0];
    int outHeight = static_cast<int>(out2D.size());
    int outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

    std::cout << "\nFinal Decrypted Convolution Output:\n";
    for (int y = 0; y < outHeight; y++) {
        for (int x = 0; x < outWidth; x++) {
            double val = he.decrypt(out2D[y][x]);
            std::cout << std::setw(6) << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
