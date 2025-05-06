/**
 * @file Neural_Layer_Skeleton.h
 * @brief Defines the core neural network layer structure
 * @author Rakib
 * @date 2025-02-14
 */

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_SKELETON_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_SKELETON_H

#include <string>
#include <iostream>
#include <stdexcept>
#include <MatrixLibrary.h>

// Forward declarations
class Neural_Block;
class Train_Block_by_Backpropagation;

/**
 * @enum ActivationType
 * @brief Supported activation functions for neural network layers
 */
enum class ActivationType {
    LINEAR,     ///< Linear activation: f(x) = x
    RELU,       ///< Rectified Linear Unit: f(x) = max(0, x)
    SIGMOID,    ///< Sigmoid activation: f(x) = 1/(1+e^(-x))
    TANH,       ///< Hyperbolic tangent: f(x) = tanh(x)
    LEAKY_RELU, ///< Leaky ReLU: f(x) = x if x > 0, else αx (α small)
    SWISH       ///< Swish activation: f(x) = x * sigmoid(x)
};

/**
 * @class Neural_Layer_Skeleton
 * @brief Represents a layer in a neural network with configurable activation function
 *
 * This class encapsulates a single neural network layer, including its weights,
 * biases, inputs, and activation functions. It enforces proper initialization
 * by requiring a valid neuron count at construction time.
 */
class Neural_Layer_Skeleton {
public:
    /**
     * @brief Creates a neural layer with specified number of neurons and activation
     *
     * @param neuron_count Number of neurons in this layer
     * @param activation_type Type of activation function to use (defaults to LINEAR)
     * @throws std::invalid_argument If number of neurons is not positive
     */
    Neural_Layer_Skeleton(int neuron_count,
                          ActivationType activation_type = ActivationType::LINEAR);

    /**
     * @brief Copy constructor - performs deep copy of all matrices
     *
     * @param other The layer to copy from
     */
    Neural_Layer_Skeleton(const Neural_Layer_Skeleton& other);

    /**
     * @brief Copy assignment operator - performs deep copy of all matrices
     *
     * @param other The layer to copy from
     * @return Reference to this layer after copying
     */
    Neural_Layer_Skeleton& operator=(const Neural_Layer_Skeleton& other);

    /**
     * @brief Move constructor - efficiently transfers resources
     *
     * @param other The layer to move from
     */
    Neural_Layer_Skeleton(Neural_Layer_Skeleton&& other) noexcept;

    /**
     * @brief Move assignment operator - efficiently transfers resources
     *
     * @param other The layer to move from
     * @return Reference to this layer after moving
     */
    Neural_Layer_Skeleton& operator=(Neural_Layer_Skeleton&& other) noexcept;

    /**
     * @brief Default destructor is fine since Matrix class handles cleanup correctly
     */
    ~Neural_Layer_Skeleton() = default;

    /**
     * @brief Gets the number of neurons in this layer
     *
     * @return The number of neurons
     */
    int get_neuron_count() const;

    /**
     * @brief Gets the activation function used by this layer
     *
     * @return The activation function type
     */
    ActivationType get_activation_type() const;

    /**
     * @brief Displays information about this layer to the console
     *
     * Prints the number of neurons and activation function type in a
     * formatted layout for easy readability.
     */
    void show_layer_info() const;

private:
    // Matrix representations of layer components
    Matrix input_matrix;             ///< Input values to this layer
    Matrix weights_matrix;           ///< Weights connecting inputs to neurons
    Matrix bias_matrix;              ///< Bias values for each neuron
    Matrix pre_activation_tensor;    ///< Values before activation function
    Matrix post_activation_tensor;   ///< Values after activation function (layer output)

    // Layer configuration
    int neurone_number_current_layer; ///< Number of neurons in this layer
    ActivationType activationType;    ///< Activation function for this layer

    /**
     * @brief Converts activation type to string representation
     *
     * @param activation_type The activation type to convert
     * @return String representation of the activation type
     * @throws std::invalid_argument If activation type is invalid
     */
    static std::string ActivationTypeToString(ActivationType activation_type);

    // Friend classes that need direct access to internal matrices
    friend class Neural_Block;
    friend class Train_Block_by_Backpropagation;
};

#endif // _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_SKELETON_H