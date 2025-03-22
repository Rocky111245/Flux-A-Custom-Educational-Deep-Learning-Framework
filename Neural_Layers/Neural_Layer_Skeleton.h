//
// Created by rakib on 14/2/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_SKELETON_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_SKELETON_H
#include <string>
#include <iostream>
#include <stdexcept>
#include <MatrixLibrary.h>


// Forward declarations of dependencies
class MatrixLibrary;  // If needed by the implementation

// Enumeration for supported activation functions
enum class ActivationType {
    LINEAR,
    RELU,
    SIGMOID,
    TANH,
    LEAKY_RELU,
    SWISH
};

class Neural_Layer_Skeleton {
public:
    // Constructors
    Neural_Layer_Skeleton() = default;

    // Parameterized constructor with optional activation
    Neural_Layer_Skeleton(int neurone_number_in_current_layer,
                          ActivationType activationType = ActivationType::LINEAR);

    // Public member functions
    void show_layer_info() const;
    //Deep Copying
    Neural_Layer_Skeleton& operator=(const Neural_Layer_Skeleton& other);

    // Getter methods
    int get_neuron_count() const;
    ActivationType get_activation_type() const;



private:
    // Private member variables. Each neurone layer must have these.To be modifiable later.
    Matrix input_matrix;
    Matrix weights_matrix;
    Matrix bias_matrix;
    Matrix pre_activation_tensor;
    Matrix post_activation_tensor;
    Matrix output_matrix;
    int neurone_number_current_layer;
    ActivationType activationType;


    // Private helper functions
    static std::string ActivationTypeToString(ActivationType activation_type);


    friend class Neural_Block;
    friend class Train_Block_by_Backpropagation;
};
#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_SKELETON_H
