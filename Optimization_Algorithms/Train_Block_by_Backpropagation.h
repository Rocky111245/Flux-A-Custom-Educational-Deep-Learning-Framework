

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H

#include "Neural_Layers/Neural_Layer_Skeleton.h"
#include "Neural_Blocks/Neural_Blocks.h"
#include <MatrixLibrary.h>
#include "Loss_Functions/Loss_Functions.h"
#include "Utility_Functions/Utility_Functions.h"
#include <vector>
#include <any>
#include <iomanip>

class Train_Block_by_Backpropagation {
public:

    // Caching structures for layer information
    struct layer_information_cache {
        Matrix input_matrix;
        Matrix weights_matrix;
        Matrix bias_matrix;
        Matrix pre_activation_tensor;
        Matrix post_activation_tensor;

        ActivationType activation_type;
    };

    // Intermediate calculations for backpropagation. Computational artifacts needed specifically for backpropagation
    struct layer_intermediate_cache {
        // Forward pass
        Matrix input_values;          // X or a_prev (input to this layer)
        Matrix pre_activation_tensor; // z values
        Matrix post_activation_tensor;    // a values (output from this layer)

        // Activation derivatives
        Matrix da_dz; // g'(z)

        // Backward pass
        Matrix dL_dz;                 // Gradient of loss w.r.t. pre-activations
        Matrix dL_dW;                 // Gradient of loss w.r.t. weights
        Matrix dL_db;                 // Gradient of loss w.r.t. biases
        Matrix dL_da_upper_layer;     // Gradient to pass to lower layer
        Matrix dL_dy;                 // Incoming gradient from the next layer
        // Cached matrices for efficiency
        Matrix W_transposed;          // Cached transposed weight matrix
        Matrix I_transposed;          // Cached transposed input matrix
        Matrix y_pred;
        Matrix y_true;

        ActivationType activation_type;


    };

    // Intermediate calculations for backpropagation. Computational artifacts needed specifically for backpropagation
    struct output_layer_cache {
        Matrix y_pred;
        Matrix y_true;
    };


    // Constructor
    Train_Block_by_Backpropagation(Neural_Block &neural_block, int iterations, float learning_rate = 0.01f);

    // Public methods
    void Train_by_Backpropagation();
    std::vector<layer_information_cache> Get_Layer_Information() const;
    std::vector<layer_intermediate_cache> Get_Intermediate_Layer_Information() const;


    //Getter methods
    int Get_Block_Size() const;
    Matrix Get_Predictions() const;



private:
    // References and parameters
    Neural_Block &neural_block;
    float learning_rate;
    int iterations;
    float cost=0.0f;




    // Storage for layer information and intermediate calculations
    std::vector<layer_information_cache> layer_information;
    std::vector<layer_intermediate_cache> intermediate_matrices;
    output_layer_cache output_layer_matrices;

    // Private helper methods
    void Populate_Layer_Information();
    void Create_Layer_Intermediate_Matrices();
    void ComputeOutputLayerGradients();
    void ComputeHiddenLayerGradients();
    void UpdateLayerParameters();
    void CalculateLossGradient(Matrix& gradient,const Matrix& predicted, const Matrix& target);
    void CalculateActivationDerivative(Matrix& derivatives,const Matrix& z,ActivationType activation_type) ;
    void UpdateLayerInformation();



};



#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TRAIN_BLOCK_BY_BACKPROPAGATION_H