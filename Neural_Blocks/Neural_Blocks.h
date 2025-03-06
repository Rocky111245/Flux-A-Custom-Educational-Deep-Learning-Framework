//
// Created by rakib on 14/2/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H

#include "Neural_Layers/Neural_Layer_Skeleton.h"
#include "Neural_Blocks.h"
#include "Neural Network Framework.h"
#include <MatrixLibrary.h>
#include "Loss Functions/Loss_Functions.h"
#include <iostream>

// Enumeration for supported activation functions


// Forward declaration of Neural_Block class
class Neural_Block {
public:
    // Constructors
    Neural_Block() = default;

    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list);
    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list,
                 LossFunction loss_function, Matrix& output_matrix);
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list);
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list,
                 LossFunction loss_function, Matrix& output_matrix);
    Neural_Block(Matrix &input_matrix, Matrix user_weights_matrix);
    Neural_Block(Matrix &input_matrix, ActivationType activation_function);

    // Public Methods
    void Forward_Pass_With_Activation();
    void Connect_With(Neural_Block& block2);
    void Calculate_Block_Loss();
    LossFunction Get_Block_Loss_Type() const;
    int Get_Block_Size() const;
    Neural_Layer_Skeleton Get_Layers(int layer_number) const;
    float Get_Loss() const;
    Matrix Get_Output_Matrix() const;



private:
    // Private Data Members
    Matrix input_matrix;
    Matrix output_matrix;
    LossFunction lossFunction;
    float loss;
    std::vector<Neural_Layer_Skeleton> layers;

    // Private Helper Methods
    void Compute_PreActivation_Matrix(Matrix &input_matrix_internal, Matrix &weights_matrix_internal,
                                      Matrix &bias_matrix_internal, Matrix &pre_activation_tensor_internal);
    void Compute_PostActivation_Matrix(Matrix &pre_activation_tensor_internal,
                                       Matrix &post_activation_tensor_internal, ActivationType activation_function_internal);
    void Apply_Activation(Matrix &pre_activation_tensor_internal, Matrix &post_activation_tensor_internal,
                          ActivationType activation_function);
    void Construct_Matrices();
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H
