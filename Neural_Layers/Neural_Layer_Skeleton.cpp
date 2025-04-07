/**
 * @file Neural_Layer_Skeleton.cpp
 * @brief Implementation of the neural network layer class
 * @author Rakib
 * @date 2025-02-14
 */

#include "Neural_Layer_Skeleton.h"

Neural_Layer_Skeleton::Neural_Layer_Skeleton(int neuron_count,
                                             ActivationType activation_type)
        : neurone_number_current_layer(neuron_count)
        , activationType(activation_type) {
    // Validate that the number of neurons is positive to ensure a valid layer structure
    if (neuron_count <= 0) {
        throw std::invalid_argument("Number of neurons must be greater than zero");
    }

    // Matrices will be properly sized later when the network is constructed
    // They are left uninitialized here since their dimensions depend on the network structure
}

Neural_Layer_Skeleton::Neural_Layer_Skeleton(const Neural_Layer_Skeleton& other)
        : input_matrix(other.input_matrix),
          weights_matrix(other.weights_matrix),
          bias_matrix(other.bias_matrix),
          pre_activation_tensor(other.pre_activation_tensor),
          post_activation_tensor(other.post_activation_tensor),
          neurone_number_current_layer(other.neurone_number_current_layer),
          activationType(other.activationType) {
    // All matrices are deep-copied through their copy constructors
}

Neural_Layer_Skeleton& Neural_Layer_Skeleton::operator=(const Neural_Layer_Skeleton& other) {
    if (this != &other) { // Prevent self-assignment
        // Copy fundamental types
        neurone_number_current_layer = other.neurone_number_current_layer;
        activationType = other.activationType;

        // Deep copy matrices
        input_matrix = other.input_matrix;
        weights_matrix = other.weights_matrix;
        bias_matrix = other.bias_matrix;
        pre_activation_tensor = other.pre_activation_tensor;
        post_activation_tensor = other.post_activation_tensor;
    }
    return *this;
}

Neural_Layer_Skeleton::Neural_Layer_Skeleton(Neural_Layer_Skeleton&& other) noexcept
        : input_matrix(std::move(other.input_matrix)),
          weights_matrix(std::move(other.weights_matrix)),
          bias_matrix(std::move(other.bias_matrix)),
          pre_activation_tensor(std::move(other.pre_activation_tensor)),
          post_activation_tensor(std::move(other.post_activation_tensor)),
          neurone_number_current_layer(other.neurone_number_current_layer),
          activationType(other.activationType) {
    // No need to reset other's primitive members since it will be destroyed
}

Neural_Layer_Skeleton& Neural_Layer_Skeleton::operator=(Neural_Layer_Skeleton&& other) noexcept {
    if (this != &other) { // Prevent self-assignment
        // Move matrices efficiently
        input_matrix = std::move(other.input_matrix);
        weights_matrix = std::move(other.weights_matrix);
        bias_matrix = std::move(other.bias_matrix);
        pre_activation_tensor = std::move(other.pre_activation_tensor);
        post_activation_tensor = std::move(other.post_activation_tensor);

        // Copy primitive types
        neurone_number_current_layer = other.neurone_number_current_layer;
        activationType = other.activationType;
    }
    return *this;
}

void Neural_Layer_Skeleton::show_layer_info() const {
    // Format and display the layer configuration details in a tree-like structure
    std::cout << "Layer Configuration:" << std::endl;
    std::cout << "├── Neurons: " << neurone_number_current_layer << std::endl;
    std::cout << "└── Activation: " << ActivationTypeToString(activationType) << std::endl;
    std::cout << std::endl;
}

int Neural_Layer_Skeleton::get_neuron_count() const {
    return neurone_number_current_layer;
}

ActivationType Neural_Layer_Skeleton::get_activation_type() const {
    return activationType;
}

std::string Neural_Layer_Skeleton::ActivationTypeToString(ActivationType activation_type) {
    // Convert enum value to its string representation for display and logging
    switch (activation_type) {
        case ActivationType::RELU:
            return "RELU";
        case ActivationType::SIGMOID:
            return "SIGMOID";
        case ActivationType::TANH:
            return "TANH";
        case ActivationType::LEAKY_RELU:
            return "LEAKY_RELU";
        case ActivationType::SWISH:
            return "SWISH";
        case ActivationType::LINEAR:
            return "LINEAR";
        default:
            throw std::invalid_argument("Invalid activation type");
    }
}