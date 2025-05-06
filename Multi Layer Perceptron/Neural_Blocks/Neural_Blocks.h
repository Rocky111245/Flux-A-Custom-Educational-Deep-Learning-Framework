/**
 * @file Neural_Blocks.h
 * @brief Defines the Neural_Block class which represents a collection of neural network layers
 * @author Rakib
 * @date 2025-02-14
 */

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H

#include "Multi Layer Perceptron/Neural_Layers/Neural_Layer_Skeleton.h"
#include <MatrixLibrary.h>
#include "Loss_Functions/Loss_Functions.h"
#include <cassert>
#include <iostream>
#include "Matrix_Dimension_Checks/Matrix_Assertion_Checks.h"
#include "Utility_Functions/Utility_Functions.h"

// Forward declarations
class Matrix;
enum class ActivationType;
enum class LossFunction;

/**
 * @class Neural_Block
 * @brief Represents a block of neural network layers that can be connected to form a complete network
 *
 * The Neural_Block class provides a flexible way to construct neural networks
 * by grouping layers and connecting blocks together. This enables modular network
 * design where blocks can be created separately and then combined. A block can be:
 * 1. An initial block with input but no output
 * 2. A middle block with neither input nor output explicitly defined
 * 3. A final block with loss function and target output
 * 4. A complete standalone network with input, layers, loss function, and target output
 */
class Neural_Block {
public:
    //===========================================================================
    // Constructors & Destructor
    //===========================================================================

    /**
     * @brief Default constructor creates an empty neural block
     */
    Neural_Block() = default;

    /**
     * @brief Creates an initial block with input matrix and layers
     *
     * @param input_matrix The input data for the network
     * @param layer_list List of neural layers in this block
     */
    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list);

    /**
     * @brief Creates a middle block with only layers defined
     *
     * @param layer_list List of neural layers in this block
     */
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list);

    /**
     * @brief Creates a final block with layers, loss function, and target output
     *
     * @param layer_list List of neural layers in this block
     * @param loss_function The loss function to use for training
     * @param output_matrix The target output data for training
     */
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list,
                 LossFunction loss_function, Matrix& output_matrix);

    /**
     * @brief Creates a complete network block with input, layers, loss function, and target output
     *
     * @param input_matrix The input data for the network
     * @param layer_list List of neural layers in this block
     * @param loss_function The loss function to use for training
     * @param target_matrix The target output data for training
     */
    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list,
                 LossFunction loss_function, Matrix& target_matrix);

    // Added this constructor to accept vectors instead of initializer_list.
    // This constructor is necessary to make it work for the web assembly.
    Neural_Block(Matrix& input_matrix, const std::vector<Neural_Layer_Skeleton>& layer_vector,
                 LossFunction loss_function, Matrix& target_matrix);

    /**
     * @brief Copy constructor - performs deep copy of all components
     *
     * @param other The neural block to copy from
     */
    Neural_Block(const Neural_Block& other);

    /**
     * @brief Move constructor - efficiently transfers ownership of resources
     *
     * @param other The neural block to move from
     */
    Neural_Block(Neural_Block&& other) noexcept;

    /**
     * @brief Copy assignment operator - performs deep copy of all components
     *
     * @param other The neural block to copy from
     * @return Reference to this neural block after copying
     */
    Neural_Block& operator=(const Neural_Block& other);

    /**
     * @brief Move assignment operator - efficiently transfers ownership of resources
     *
     * @param other The neural block to move from
     * @return Reference to this neural block after moving
     */
    Neural_Block& operator=(Neural_Block&& other) noexcept;

    /**
     * @brief Destructor - cleans up resources
     */
    ~Neural_Block();

    //===========================================================================
    // Public Methods
    //===========================================================================

    /**
     * @brief Performs forward pass through all layers in the block
     *
     * Computes the output of each layer in sequence, passing the output of one
     * layer as input to the next. Also updates the loss value if the block
     * has a target matrix and loss function.
     */
    void Forward_Pass_With_Activation();

    /**
     * @brief Connects this block with another block to form a larger network
     *
     * @param block2 The block to connect to this one
     * @return Reference to the combined block
     * @throws std::runtime_error If the blocks cannot be connected due to incompatible structure
     */
    Neural_Block& Connect_With(Neural_Block& block2);

    //===========================================================================
    // Getters
    //===========================================================================

    /**
     * @brief Checks if the block is complete with input, layers, and output
     *
     * @return true if the block is complete, false otherwise
     */
    bool Get_Block_Status() const;

    /**
     * @brief Gets the loss function type used by this block
     *
     * @return The loss function enumeration value
     */
    LossFunction Get_Block_Loss_Type() const;

    /**
     * @brief Gets the number of layers in this block
     *
     * @return Number of layers
     */
    int Get_Block_Size() const;

    /**
     * @brief Gets a reference to a specific layer for full access
     *
     * @param layer_number Index of the layer (0-based)
     * @return Reference to the neural layer
     * @throws std::out_of_range If layer_number is invalid
     */
    Neural_Layer_Skeleton& Set_Block_Layers(int layer_number);

    /**
     * @brief Gets the current loss value after forward pass
     *
     * @return The loss value
     */
    float Get_Block_Loss() const;

    /**
     * @brief Gets the input matrix
     *
     * @return Copy of the input matrix
     */
    Matrix Get_Block_Input_Matrix() const;

    /**
     * @brief Gets a reference to the target matrix
     *
     * @return Const reference to the target matrix
     */
    const Matrix& Get_Block_Target_Matrix() const;

    /**
     * @brief Gets a reference to the weights matrix for a specific layer
     *
     * @param layer_number Index of the layer (0-based)
     * @return Reference to the weights matrix
     * @throws std::out_of_range If layer_number is invalid
     */
    Matrix& Get_Block_Weights_Matrix(int layer_number);

    /**
     * @brief Gets a reference to the bias matrix for a specific layer
     *
     * @param layer_number Index of the layer (0-based)
     * @return Reference to the bias matrix
     * @throws std::out_of_range If layer_number is invalid
     */
    Matrix& Get_Block_Bias_Matrix(int layer_number);

    //===========================================================================
    // Setters
    //===========================================================================

    /**
     * @brief Gets a mutable reference to a layer for modification
     *
     * @param layer_number Index of the layer (0-based)
     * @return Reference to the neural layer
     * @throws std::out_of_range If layer_number is invalid
     */
    Neural_Layer_Skeleton& Set_Layers(int layer_number);

private:
    //===========================================================================
    // Private Data Members
    //===========================================================================

    Matrix input_matrix;                ///< Input data for this block
    Matrix target_matrix;               ///< Target output data for training
    LossFunction lossFunction;          ///< Loss function for evaluating predictions
    float loss = 0.0f;                  ///< Current loss value
    std::vector<Neural_Layer_Skeleton> layers; ///< Layers contained in this block
    bool input_matrix_constructed = false;   ///< Flag indicating if input is initialized
    bool target_matrix_constructed = false;  ///< Flag indicating if target is initialized
    bool loss_function_constructed = false;  ///< Flag indicating if loss function is set

    //===========================================================================
    // Private Helper Methods
    //===========================================================================

    /**
     * @brief Initializes matrices for all layers with correct dimensions
     *
     * Creates and initializes weights, biases, and activation matrices for all
     * layers in the block. Uses Xavier initialization for weights.
     */
    void Construct_Matrices();

    /**
     * @brief Calculates loss for this block using the configured loss function
     *
     * Computes the loss between the output of the last layer and the target matrix.
     */
    void Calculate_Block_Loss();

    /**
     * @brief Computes pre-activation values for a layer
     *
     * Calculates z = x*W + b where x is input, W is weights, and b is bias.
     *
     * @param input_matrix_internal Input values to the layer
     * @param weights_matrix_internal Weights matrix for the layer
     * @param bias_matrix_internal Bias matrix for the layer
     * @param pre_activation_tensor_internal Output matrix for pre-activation values
     */
    void Compute_PreActivation_Matrix(Matrix& input_matrix_internal,
                                      Matrix& weights_matrix_internal,
                                      Matrix& bias_matrix_internal,
                                      Matrix& pre_activation_tensor_internal);

    /**
     * @brief Applies activation function to pre-activation values
     *
     * @param pre_activation_tensor_internal Pre-activation input values
     * @param post_activation_tensor_internal Output matrix for post-activation values
     * @param activation_function_internal Type of activation function to apply
     */
    void Compute_PostActivation_Matrix(Matrix& pre_activation_tensor_internal,
                                       Matrix& post_activation_tensor_internal,
                                       ActivationType activation_function_internal);

    /**
     * @brief Applies activation function to an entire matrix
     *
     * @param result Output matrix for activated values
     * @param input Input matrix of values to activate
     * @param activation_type Type of activation function to apply
     * @throws std::invalid_argument If matrices have incompatible dimensions
     */
    void Apply_Activation_Function_To_Matrix(Matrix& result, const Matrix& input,
                                             ActivationType activation_type);

    /**
     * @brief Gets a reference to the pre-activation matrix for a specific layer
     *
     * @param layer_number Index of the layer (0-based)
     * @return Reference to the pre-activation matrix
     * @throws std::out_of_range If layer_number is invalid
     */
    Matrix& Get_Block_Pre_Activation_Matrix(int layer_number);

    /**
     * @brief Gets a reference to the post-activation matrix for a specific layer
     *
     * @param layer_number Index of the layer (0-based)
     * @return Reference to the post-activation matrix
     * @throws std::out_of_range If layer_number is invalid
     */
    Matrix& Get_Block_Post_Activation_Matrix(int layer_number);
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H