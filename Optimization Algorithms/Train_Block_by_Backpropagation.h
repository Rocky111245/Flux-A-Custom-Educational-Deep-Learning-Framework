//
// Created by rakib on 16/2/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H

#include "Neural_Layers/Neural_Layer_Skeleton.h"
#include "Neural_Blocks/Neural_Blocks.h"
#include <MatrixLibrary.h>
#include "Loss Functions/Loss_Functions.h"
#include <vector>

class Train_Block_by_Backpropagation {
public:
    // Constructor
    Train_Block_by_Backpropagation(Neural_Block &neural_block, int iterations, float learning_rate = 0.01f);

    // Public methods
    void Train_by_Backpropagation(int iterations);

private:
    // References and parameters
    Neural_Block &neural_block;
    float learning_rate;
    int iterations;

    // Caching structures for layer information
    struct layer_information_cache {
        Matrix input_matrix;
        Matrix weights_matrix;
        Matrix bias_matrix;
        Matrix pre_activation_values;
        Matrix activation_outputs;
        float cost;
        ActivationType activation_type;
    };

    // Intermediate calculations for backpropagation
    struct layer_intermediate_cache {
        // Forward pass
        Matrix input_values;          // X or a_prev (input to this layer)
        Matrix pre_activation_values; // z values
        Matrix activation_outputs;    // a values (output from this layer)

        // Activation derivatives
        Matrix activation_derivatives; // g'(z)

        // Backward pass
        Matrix dL_dz;                 // Gradient of loss w.r.t. pre-activations
        Matrix dL_dW;                 // Gradient of loss w.r.t. weights
        Matrix dL_db;                 // Gradient of loss w.r.t. biases
        Matrix dL_da_prev;            // Gradient to pass to previous layer
        Matrix dL_dy;                 // Incoming gradient from the next layer

        ActivationType activation_type;

        // Cached matrices for efficiency
        Matrix W_transposed;          // Cached transposed weight matrix
        Matrix I_transposed;          // Cached transposed input matrix
    };

    // Storage for layer information and intermediate calculations
    std::vector<layer_information_cache> layer_information;
    std::vector<layer_intermediate_cache> intermediate_matrices;

    // Private helper methods
    void Populate_Layer_Information();
    void Create_Layer_Intermediate_Matrices();
    void ComputeOutputLayerGradients();
    void ComputeHiddenLayerGradients();
    void UpdateLayerParameters();
    void CalculateLossGradient(const Matrix& predicted, const Matrix& target, Matrix& gradient);
    void CalculateActivationDerivative(const Matrix& z, Matrix& derivatives, ActivationType activation_type);
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H