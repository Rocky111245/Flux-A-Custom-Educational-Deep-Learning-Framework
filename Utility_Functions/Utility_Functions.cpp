#include "Utility_Functions.h"
#include "Optimization_Algorithms/Train_Block_by_Backpropagation.h"
#include "Neural_Layers/Neural_Layer_Skeleton.h" // For ActivationType

void DisplayNeuralLayerMatrices(const Train_Block_by_Backpropagation& trained_block) {
    try {
        int block_size = trained_block.Get_Block_Size();
        std::vector<Train_Block_by_Backpropagation::layer_information_cache> layer_information =
                trained_block.Get_Layer_Information();

        std::cout << "==== Neural Network Architecture ====" << std::endl;
        std::cout << "Total layers: " << block_size << std::endl << std::endl;

        for (int i = 0; i < block_size; i++) {
            std::cout << "=== Layer " << i << " (" << (i == 0 ? "Input" :
                                                       (i == block_size-1 ? "Output" : "Hidden")) << ") ===" << std::endl;

            // Display matrix dimensions with consistent formatting
            std::cout << "  Input Matrix: [" << layer_information[i].input_matrix.rows()
                      << " X " << layer_information[i].input_matrix.columns() << "]" << std::endl;
            std::cout << "  Weights Matrix: [" << layer_information[i].weights_matrix.rows()
                      << " X  " << layer_information[i].weights_matrix.columns() << "]" << std::endl;
            std::cout << "  Bias Matrix: [" << layer_information[i].bias_matrix.rows()
                      << " X"
                         " " << layer_information[i].bias_matrix.columns() << "]" << std::endl;
            std::cout << "  Pre-Activation Matrix: [" << layer_information[i].pre_activation_tensor.rows()
                      << " X"
                         " " << layer_information[i].pre_activation_tensor.columns() << "]" << std::endl;
            std::cout << "  Post-Activation Matrix: [" << layer_information[i].post_activation_tensor.rows()
                      << " X"
                         " " << layer_information[i].post_activation_tensor.columns() << "]" << std::endl;

            // Add activation type information
            std::cout << "  Activation: " << GetActivationTypeString(layer_information[i].activation_type) << std::endl;
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error displaying neural layer matrices: " << e.what() << std::endl;
    }
}

void DisplayBackpropagationGradients(const Train_Block_by_Backpropagation& trained_block) {
    try {
        int block_size = trained_block.Get_Block_Size();
        std::vector<Train_Block_by_Backpropagation::layer_intermediate_cache> intermediate_matrices =
                trained_block.Get_Intermediate_Layer_Information();

        std::cout << "==== Backpropagation Gradient Matrices ====" << std::endl;
        std::cout << "Total layers: " << block_size << std::endl << std::endl;

        for (int i = 0; i < block_size; i++) {
            if (i < block_size - 1) {
                std::cout << "=== Layer " << i << " Gradients ===" << std::endl;

                std::cout << "  Activation Derivative (da/dz): [" << intermediate_matrices[i].da_dz.rows()
                          << " X"
                             " " << intermediate_matrices[i].da_dz.columns() << "]" << std::endl;
                std::cout << "  Loss Gradient wrt Pre-Activation (dL/dz): [" << intermediate_matrices[i].dL_dz.rows()
                          << " X"
                             " " << intermediate_matrices[i].dL_dz.columns() << "]" << std::endl;
                std::cout << "  Loss Gradient wrt Weights (dL/dW): [" << intermediate_matrices[i].dL_dW.rows()
                          << " X"
                             " " << intermediate_matrices[i].dL_dW.columns() << "]" << std::endl;
                std::cout << "  Loss Gradient wrt Biases (dL/db): [" << intermediate_matrices[i].dL_db.rows()
                          << " X"
                             " " << intermediate_matrices[i].dL_db.columns() << "]" << std::endl;
                std::cout << "  Loss Gradient for Upper Layer (dL/da): [" << intermediate_matrices[i].dL_da_upper_layer.rows()
                          << " X"
                             " " << intermediate_matrices[i].dL_da_upper_layer.columns() << "]" << std::endl;
                std::cout << "  Loss Gradient wrt Output (dL/dy): [" << intermediate_matrices[i].dL_dy.rows()
                          << " X"
                             " " << intermediate_matrices[i].dL_dy.columns() << "]" << std::endl;
                std::cout << "  Transposed Weights Matrix: [" << intermediate_matrices[i].W_transposed.rows()
                          << " X"
                             " " << intermediate_matrices[i].W_transposed.columns() << "]" << std::endl;
                std::cout << "  Transposed Input Matrix: [" << intermediate_matrices[i].I_transposed.rows()
                          << " X"
                             " " << intermediate_matrices[i].I_transposed.columns() << "]" << std::endl;
            } else {
                std::cout << "=== Output Layer " << i << " ===" << std::endl;
                std::cout << "  Predicted Values: [" << intermediate_matrices[i].y_pred.rows()
                          << " X"
                             " " << intermediate_matrices[i].y_pred.columns() << "]" << std::endl;
                std::cout << "  Target Values: [" << intermediate_matrices[i].y_true.rows()
                          << " X"
                             " " << intermediate_matrices[i].y_true.columns() << "]" << std::endl;


            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error displaying backpropagation gradients: " << e.what() << std::endl;
    }
}

// Helper function to convert activation type to string
std::string GetActivationTypeString(ActivationType type) {
    switch (type) {
        case ActivationType::RELU: return "ReLU";
        case ActivationType::SIGMOID: return "Sigmoid";
        case ActivationType::TANH: return "Tanh";
        case ActivationType::LEAKY_RELU: return "Leaky ReLU";
        case ActivationType::SWISH: return "Swish";
        case ActivationType::LINEAR: return "Linear";
        default: return "Unknown";
    }
}