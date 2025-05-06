/**
 * @file Train_Block_by_Backpropagation.h
 * @brief Defines backpropagation training algorithm for neural network blocks
 * @author Rakib
 * @date 2025-03-28
 */

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H

#include "Multi Layer Perceptron/Neural_Layers/Neural_Layer_Skeleton.h"
#include "Multi Layer Perceptron/Neural_Blocks/Neural_Blocks.h"
#include <MatrixLibrary.h>
#include "Loss_Functions/Loss_Functions.h"
#include "Utility_Functions/Utility_Functions.h"
#include <vector>
#include <any>
#include <iomanip>

// Forward declarations
class Neural_Block;
class Matrix;
enum class ActivationType;

/**
 * @class Train_Block_by_Backpropagation
 * @brief Implements backpropagation algorithm for training neural network blocks
 *
 * This class provides functionality to train Neural_Block instances using the
 * backpropagation algorithm. It handles gradient computation, parameter updates,
 * and tracks training progress. The implementation uses a pointer-based design
 * for efficiency and to maintain tight coupling between forward and backward passes.
 */
class Train_Block_by_Backpropagation {
public:
    /**
     * @struct layer_intermediate_cache
     * @brief Caches intermediate values needed for backpropagation
     *
     * Pointer-based design is used for the following reasons:
     * 1) Forward and backward passes are tightly coupled. Storing direct pointers avoids
     *    redundant data passing between functions, reducing memory overhead, code bloat,
     *    and improving clarity.
     * 2) Memory ownership is handled by the NeuralBlock class. This struct only references
     *    existing matrices and does not manage memory lifetime.
     * 3) Simplifies code design by enabling direct in-place updates during training.
     */
    struct layer_intermediate_cache {
        // ----- Forward Pass -----
        Matrix* input_values = nullptr;          ///< Input to the layer (X or a_prev)
        Matrix* pre_activation_tensor = nullptr; ///< Linear combination before activation (Z)
        Matrix* post_activation_tensor = nullptr; ///< Output after activation (A)
        Matrix* weights_matrix = nullptr;        ///< Cached pointer to layer's weights
        Matrix* bias_matrix = nullptr;           ///< Cached pointer to layer's biases
        Matrix* y_pred;                          ///< Model predictions (only for output layer)

        // ----- Cached Matrices for Efficiency -----
        Matrix W_transposed;                     ///< Cached transpose of weights (Wᵀ)
        Matrix I_transposed;                     ///< Cached transpose of inputs (Xᵀ)

        // ----- Activation Derivatives -----
        Matrix da_dz;                            ///< Derivative of activation function (g'(z))

        // ----- Backward Pass -----
        Matrix dL_dz;                            ///< Gradient of loss w.r.t. pre-activation values
        Matrix dL_dW;                            ///< Gradient of loss w.r.t. weights
        Matrix dL_db;                            ///< Gradient of loss w.r.t. biases
        Matrix dL_da_upper;                      ///< Gradient to propagate to previous layer
        Matrix dL_dy;                            ///< Incoming gradient from the next layer

        // ----- Output Layer Only -----
        Matrix y_true;                           ///< Ground truth labels (only for output layer)

        // ----- Layer Meta Information -----
        ActivationType activation_type;          ///< Activation function used in the layer
    };

    /**
     * @brief Constructs a trainer for the given neural block
     *
     * Initializes all training resources and immediately starts the training process
     * for the specified number of iterations.
     *
     * @param neural_block The neural network block to train
     * @param iterations Number of training iterations to perform
     * @param learning_rate Step size for gradient descent updates (default: 0.01)
     */
    Train_Block_by_Backpropagation(Neural_Block &neural_block, int iterations, float learning_rate = 0.01f);


    /**
     * @brief Performs backpropagation training for the specified number of iterations
     *
     * This method runs forward pass, gradient computation, and parameter updates
     * for each iteration, tracking and reporting the loss.
     */
    void Train_by_Backpropagation();
    void Train_by_Backpropagation_One_Iteration();


    /**
     * @brief Gets all intermediate gradient matrices for inspection
     *
     * @return Vector of all intermediate caches used during training
     */
    std::vector<layer_intermediate_cache> Get_Intermediate_Layer_Information() const;

    /**
     * @brief Gets the number of layers in the neural block being trained
     *
     * @return Number of layers in the neural block
     */
    int Get_Block_Size() const;

    /**
     * @brief Gets the predictions from the trained model
     *
     * @return Matrix containing model predictions (output of the last layer)
     */
    Matrix Get_Predictions() const;

private:
    // References and parameters
    Neural_Block &neural_block;     ///< Reference to the neural block being trained
    float learning_rate;            ///< Learning rate for parameter updates
    int iterations;                 ///< Number of training iterations
    float cost = 0.0f;              ///< Current loss value

    // Storage for intermediate calculations
    std::vector<layer_intermediate_cache> intermediate_matrices; ///< Cached values for each layer

    /**
     * @brief Initializes all matrices needed for backpropagation
     *
     * Sets up gradient matrices, cached transposed matrices, and pointers to
     * the neural block's matrices for efficient in-place updates.
     */
    void Initialize_Layer_Intermediate_Matrices();

    /**
     * @brief Computes gradients for all layers
     *
     * Performs the backward pass, computing gradients of the loss with respect
     * to all parameters (weights and biases) and activation values.
     */
    void ComputeAllLayerGradients();

    /**
     * @brief Updates weights and biases using computed gradients
     *
     * Applies gradient descent updates to all model parameters using the
     * computed gradients scaled by the learning rate.
     */
    void UpdateLayerParameters();

    /**
     * @brief Calculates the gradient of the loss with respect to predictions
     *
     * Computes the derivative of the loss function with respect to model outputs.
     * The specific formula depends on the loss function type.
     *
     * @param gradient Output matrix to store the computed gradient
     * @param predicted Matrix of model predictions
     * @param target Matrix of ground truth values
     */
    void CalculateLossGradient(Matrix& gradient, const Matrix& predicted, const Matrix& target);

    /**
     * @brief Calculates derivatives of activation functions
     *
     * Computes the derivative of the specified activation function with respect
     * to its inputs.
     *
     * @param derivatives Output matrix to store computed derivatives
     * @param z Matrix of pre-activation values
     * @param activation_type Type of activation function
     * @throws std::invalid_argument If matrices have incompatible dimensions
     */
    void CalculateActivationDerivative(Matrix& derivatives, const Matrix& z, ActivationType activation_type);

    /**
     * @brief Sums columns of a matrix into a single row
     *
     * Used for computing bias gradients by summing gradients across all training examples.
     *
     * @param dest Destination matrix (single row)
     * @param src Source matrix to sum
     * @throws std::invalid_argument If matrices have incompatible dimensions
     */
    void Matrix_Sum_Columns_To_One_Row(Matrix &dest, const Matrix &src);

    /**
     * @brief Updates cached information after a forward pass
     *
     * Recomputes transposed matrices and other cached values that depend
     * on the latest forward pass results.
     */
    void UpdateInformationAfterForwardPass();

};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H