// Neural_Layer_Skeleton.cpp
// Implementation file for the Neural Layer Skeleton class

#include "Neural_Layer_Skeleton.h"



// Initializes a neural layer with specified number of neurons and activation function
Neural_Layer_Skeleton::Neural_Layer_Skeleton(int neurone_number_in_current_layer,
                                             ActivationType activationType)
        : neurone_number_current_layer(neurone_number_in_current_layer)
        , activationType(activationType) {
    // Validate that the number of neurons is positive to ensure a valid layer structure
    if (neurone_number_current_layer <= 0) {
        throw std::invalid_argument("Number of neurons must be positive");
    }
}

Neural_Layer_Skeleton& Neural_Layer_Skeleton::operator=(const Neural_Layer_Skeleton& other) {
    if (this != &other) { // Prevent self-assignment
        this->neurone_number_current_layer = other.neurone_number_current_layer;
        this->activationType = other.activationType;

        // Deep copy of matrices
        this->input_matrix = other.input_matrix;
        this->weights_matrix = other.weights_matrix;
        this->bias_matrix = other.bias_matrix;
        this->pre_activation_tensor = other.pre_activation_tensor;
        this->post_activation_tensor = other.post_activation_tensor;
    }
    return *this;
}




// Public method implementation for displaying the layer configuration
void Neural_Layer_Skeleton::show_layer_info() const {
    std::cout << "Layer Configuration:" << std::endl;
    std::cout << "├── Neurons: " << neurone_number_current_layer << std::endl;
    std::cout << "└── Activation: " << ActivationTypeToString(activationType) << std::endl;
    std::cout<< std::endl;
}




// Getter method implementation for accessing the number of neurons
int Neural_Layer_Skeleton::get_neuron_count() const {
    return neurone_number_current_layer;
}

// Getter method implementation for accessing the activation type
ActivationType Neural_Layer_Skeleton::get_activation_type() const {
    return activationType;
}

// Private helper method implementation to convert activation types to string representations
std::string Neural_Layer_Skeleton::ActivationTypeToString(ActivationType activation_type) {
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