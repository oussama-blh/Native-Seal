#include <torch/script.h>  // For TorchScript model loading
#include <torch/torch.h>
#include <iostream>
#include <iomanip>  // For formatted output
#include <string>
#include <vector>
#include <unordered_map>
#include <he/he.h>
#include <convolution/convolution.h>
#include <linear/linear.h>
#include <functions/square.h>

// Structure for convolutional layer weights
struct ConvLayerWeights {
    std::vector<std::vector<std::vector<std::vector<float>>>> weights;
};

// Structure for linear layer weights
struct LinearLayerWeights {
    std::vector<std::vector<float>> weights;
};

// A structure to hold both weight and bias for one layer.
struct LayerParameters {
    std::string layerId;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> biasShape;
    bool isConv;
    ConvLayerWeights conv;
    LinearLayerWeights linear;
    std::vector<float> bias;
};

// Extract layer ID from parameter name.
std::string extractLayerId(const std::string &paramName) {
    size_t pos = paramName.find('.');
    return (pos != std::string::npos) ? paramName.substr(0, pos) : paramName;
}

// Function to extract weights and biases from a TorchScript model
std::unordered_map<std::string, LayerParameters> extractWeightsAndBiases(const std::string& modelPath) {
    std::unordered_map<std::string, LayerParameters> layerMap;
    
    try {
        torch::jit::Module module = torch::jit::load(modelPath);
        
        for (const auto& param : module.named_parameters()) {
            std::string fullName = param.name;
            torch::Tensor tensor = param.value;
            std::vector<int64_t> shape = tensor.sizes().vec();
            bool isWeight = (fullName.find("weight") != std::string::npos);
            bool isBias = (fullName.find("bias") != std::string::npos);
            std::string layerId = extractLayerId(fullName);
            
            if (layerMap.find(layerId) == layerMap.end()) {
                layerMap[layerId] = {layerId, {}, {}, false, {}, {}, {}};
            }
            LayerParameters &layer = layerMap[layerId];
            
            if (isWeight) {
                layer.weightShape = shape;
                if (tensor.dim() == 4) {
                    layer.isConv = true;
                    int channels = tensor.size(0), filters = tensor.size(1);
                    int height = tensor.size(2), width = tensor.size(3);
                    layer.conv.weights.resize(channels);
                    auto data = tensor.accessor<float, 4>();
                    for (int i = 0; i < channels; ++i) {
                        layer.conv.weights[i].resize(filters);
                        for (int j = 0; j < filters; ++j) {
                            layer.conv.weights[i][j].resize(height);
                            for (int k = 0; k < height; ++k) {
                                layer.conv.weights[i][j][k].resize(width);
                                for (int l = 0; l < width; ++l) {
                                    layer.conv.weights[i][j][k][l] = data[i][j][k][l];
                                }
                            }
                        }
                    }
                } else if (tensor.dim() == 2) {
                    layer.isConv = false;
                    int rows = tensor.size(0), cols = tensor.size(1);
                    layer.linear.weights.resize(rows);
                    auto data = tensor.accessor<float, 2>();
                    for (int i = 0; i < rows; ++i) {
                        layer.linear.weights[i].resize(cols);
                        for (int j = 0; j < cols; ++j) {
                            layer.linear.weights[i][j] = data[i][j];
                        }
                    }
                }
            } else if (isBias) {
                layer.biasShape = shape;
                layer.bias.resize(tensor.numel());
                auto data = tensor.accessor<float, 1>();
                for (size_t i = 0; i < layer.bias.size(); ++i) {
                    layer.bias[i] = data[i];
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
    }
    
    return layerMap;
}

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
    const std::string MODEL_PATH = "C:/Khbich/PFE/Implementations/NativeSEAL/models/Lenet1_traced.pt";
    
    // Extract layer weights
    std::unordered_map<std::string, LayerParameters> layerMap = extractWeightsAndBiases(MODEL_PATH);

    // Initialize CKKS encryption
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

    // auto outputEnc1;

    // Encrypt the input image
    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> inputEnc(1);
    inputEnc[0].resize(1);
    inputEnc[0][0] = he.encryptMatrix2D(inputDouble[0][0]); // Encrypt the MNIST image

    // Retrieve convolutional and linear layer parameters by index
    std::string convLayerId = "0";  // First convolutional layer
    std::string linearLayerId = "7";  // Fully connected layer

    std::vector<std::vector<std::vector<std::vector<seal::Ciphertext>>>> outputEnc1;
    // **Step 1: Initialize Convolutional Layer**
    if (layerMap.find(convLayerId) != layerMap.end()) {
        LayerParameters& convLayerParams = layerMap[convLayerId];

        if (convLayerParams.isConv) {
            std::vector<std::vector<std::vector<std::vector<double>>>> convertedWeights(
                convLayerParams.conv.weights.size()
            );

            for (size_t i = 0; i < convLayerParams.conv.weights.size(); i++) {
                convertedWeights[i].resize(convLayerParams.conv.weights[i].size());
                for (size_t j = 0; j < convLayerParams.conv.weights[i].size(); j++) {
                    convertedWeights[i][j].resize(convLayerParams.conv.weights[i][j].size());
                    for (size_t k = 0; k < convLayerParams.conv.weights[i][j].size(); k++) {
                        convertedWeights[i][j][k].resize(convLayerParams.conv.weights[i][j][k].size());
                        for (size_t l = 0; l < convLayerParams.conv.weights[i][j][k].size(); l++) {
                            convertedWeights[i][j][k][l] = static_cast<double>(convLayerParams.conv.weights[i][j][k][l]);
                        }
                    }
                }
            }
            std::vector<double> convertedBias(convLayerParams.bias.begin(), convLayerParams.bias.end());

            // Create convolutional layer
            Conv2d convLayer(
                he, convertedWeights, 
                std::make_pair(1, 1),  // Stride
                std::make_pair(1, 1),  // Padding
                convertedBias
            );

            std::cout << "Initialized Conv Layer 0!" << std::endl;

            // duration measurement
            auto start = std::chrono::high_resolution_clock::now();
    
            //  Apply first convolution layer
            outputEnc1 = convLayer(inputEnc);

            auto end = std::chrono::high_resolution_clock::now();

            // For milliseconds:
            auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Time taken: " << duration_ms.count() << " milliseconds" << std::endl;
            
            // 8) Decrypt and print the final convolution result
            // auto &out2D = outputEnc1[0][0];
            // int outHeight = static_cast<int>(out2D.size());
            // int outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

            // std::cout << "\n First Decrypted Convolution Output:\n";
            // for (int y = 0; y < outHeight; y++) {
            //     for (int x = 0; x < outWidth; x++) {
            //         double val = he.decrypt(out2D[y][x]);
            //         std::cout << std::setw(6) << val << " ";
            //     }
            //     std::cout << std::endl;
            // }
        }
    }

    //Apply SquareLayer after the first convolution
    auto squareLayer = SquareLayer(he);

    auto squaredEnc = squareLayer(outputEnc1);

    // auto &out2D = squaredEnc[0][0];
    // int outHeight = static_cast<int>(out2D.size());
    // int outWidth  = (outHeight > 0) ? static_cast<int>(out2D[0].size()) : 0;

    // std::cout << "\nFirst Decrypted Square Output:\n";
    // for (int y = 0; y < outHeight; y++) {
    //     for (int x = 0; x < outWidth; x++) {
    //         double val = he.decrypt(out2D[y][x]);
    //         std::cout << std::setw(6) << val << " ";
    //     }
    //     std::cout << std::endl;
    // }


    // **Step 2: Initialize Fully Connected (Linear) Layer**
    if (layerMap.find(linearLayerId) != layerMap.end()) {
        LayerParameters& linearLayerParams = layerMap[linearLayerId];

        if (!linearLayerParams.isConv) {
            std::vector<std::vector<double>> convertedWeights(linearLayerParams.linear.weights.size());
            for (size_t i = 0; i < linearLayerParams.linear.weights.size(); i++) {
                convertedWeights[i].resize(linearLayerParams.linear.weights[i].size());
                for (size_t j = 0; j < linearLayerParams.linear.weights[i].size(); j++) {
                    convertedWeights[i][j] = static_cast<double>(linearLayerParams.linear.weights[i][j]);
                }
            }
            std::vector<double> convertedBias(linearLayerParams.bias.begin(), linearLayerParams.bias.end());

            // Create linear layer
            LinearLayer linearLayer(he, convertedWeights, convertedBias);
            std::cout << "Initialized Linear Layer 3!" << std::endl;
        }
    }

    return 0;
}
