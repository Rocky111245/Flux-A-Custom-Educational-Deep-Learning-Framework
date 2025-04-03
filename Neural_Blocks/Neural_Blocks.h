//
// Created by rakib on 14/2/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H

#include "Neural_Layers/Neural_Layer_Skeleton.h"
#include "Neural_Blocks.h"

#include <MatrixLibrary.h>
#include "Loss_Functions/Loss_Functions.h"
#include <cassert>
#include <iostream>
#include "Matrix_Dimension_Checks/Matrix_Assertion_Checks.h"
#include "Utility_Functions/Utility_Functions.h"

// Forward declaration of Neural_Block class
class Neural_Block {
public:
    //===========================================================================
    // Constructors & Destructor
    //===========================================================================

    // Default constructor
    Neural_Block() = default;

    // Constructor for initial block with input but no output
    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list);

    // Constructor for middle blocks without predefined input
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list);

    // Constructor for final blocks with loss function and output matrix
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list,
                 LossFunction loss_function, Matrix& output_matrix);

    // Constructor for complete network blocks
    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list,
                 LossFunction loss_function, Matrix& target_matrix);

    // Copy constructor
    Neural_Block(const Neural_Block& other);

    // Move constructor
    Neural_Block(Neural_Block&& other) noexcept;

    // Copy assignment operator
    Neural_Block& operator=(const Neural_Block& other);

    // Move assignment operator
    Neural_Block& operator=(Neural_Block&& other) noexcept;

    // Destructor
    ~Neural_Block();

    //===========================================================================
    // Public Methods
    //===========================================================================

    // Perform forward pass through all layers
    void Forward_Pass_With_Activation();

    // Connect this block with another block
    Neural_Block& Connect_With(Neural_Block& block2);



    //===========================================================================
    // Getters
    //===========================================================================

    // Get status of the block completion
    bool Get_Block_Status() const;

    // Get the loss function type
    LossFunction Get_Block_Loss_Type() const;

    // Get number of layers in the block
    int Get_Block_Size() const;

    // Get a full specific layer by index. Exposes the whole layer's variables.
    const Neural_Layer_Skeleton& Get_Block_Layers(int layer_number) const;

    // Get the current loss value
    float Get_Block_Loss() const;

    // Get the input matrix
    Matrix Get_Block_Input_Matrix() const;

    // Get the output matrix
    const Matrix& Get_Block_Target_Matrix() const;


    // Get weights matrix for a specific layer
    Matrix& Get_Block_Weights_Matrix(int layer_number);

    // Get bias matrix for a specific layer
    Matrix& Get_Block_Bias_Matrix(int layer_number);

    //===========================================================================
    // Setters
    //===========================================================================

    // Get a mutable reference to a layer for modification
    Neural_Layer_Skeleton& Set_Layers(int layer_number);

private:
    //===========================================================================
    // Private Data Members
    //===========================================================================

    Matrix input_matrix;
    Matrix target_matrix;
    LossFunction lossFunction;
    float loss = 0.0f;
    std::vector<Neural_Layer_Skeleton> layers;
    bool input_matrix_constructed = false;
    bool target_matrix_constructed = false;
    bool loss_function_constructed = false;

    //===========================================================================
    // Private Helper Methods
    //===========================================================================

    // Initialize matrices for all layers
    void Construct_Matrices();

    // Calculate loss for this block
    void Calculate_Block_Loss();

    // Compute pre-activation values
    void Compute_PreActivation_Matrix(Matrix& input_matrix_internal,
                                      Matrix& weights_matrix_internal,
                                      Matrix& bias_matrix_internal,
                                      Matrix& pre_activation_tensor_internal);

    // Apply activation function to create post-activation values
    void Compute_PostActivation_Matrix(Matrix& pre_activation_tensor_internal,
                                       Matrix& post_activation_tensor_internal,
                                       ActivationType activation_function_internal);

    void Apply_Activation_Function_To_Matrix(Matrix& result, const Matrix& input,
                                             ActivationType activation_type);


    Matrix &Get_Block_Pre_Activation_Matrix(int layer_number);

    Matrix &Get_Block_Post_Activation_Matrix(int layer_number);
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H