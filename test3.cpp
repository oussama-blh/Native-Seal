#include <torch/script.h>  // For TorchScript model loading
#include <iostream>
#include <string>
#include <he/he.h>
const std::string MODEL_PATH = "C:/Khbich/PFE/Implementations/NativeSEAL/models/Lenet1_traced.pt";


// Function to create and return a CKKS instance with keys generated
CKKSPyfhel createCKKSInstance() {
    CKKSPyfhel he;
    he.generate_keys();
    he.generate_relin_keys();
    return he;  // Return the initialized CKKS instance
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

    
    try {
        //Load model
        torch::jit::Module module = torch::jit::load(MODEL_PATH);

        //Iterate through layers
        std::cout << "Model Weights:" << std::endl;
        for (const auto& param : module.named_parameters()) {
            std::cout << "Layer: " << param.name << std::endl;
            std::cout << param.value << std::endl << std::endl;
        }

    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
