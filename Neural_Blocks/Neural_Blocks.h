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

// Enumeration for supported activation functions


// Forward declaration of Neural_Block class
class Neural_Block {
public:
    // Constructors
    Neural_Block() = default;

    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list);
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list);
    Neural_Block(std::initializer_list<Neural_Layer_Skeleton> layer_list, LossFunction loss_function, Matrix& output_matrix);

    Neural_Block(Matrix& input_matrix, std::initializer_list<Neural_Layer_Skeleton> layer_list, LossFunction loss_function, Matrix& output_matrix);


    // Public Methods
    void Forward_Pass_With_Activation();
    Neural_Block& Connect_With(Neural_Block& block2);
    void Calculate_Block_Loss();

    //Getter Methods
    bool Get_Block_Status() const;
    LossFunction Get_Block_Loss_Type() const;
    int Get_Block_Size() const;
    const Neural_Layer_Skeleton &Get_Layers(int layer_number) const;
    float Get_Loss() const;
    Matrix Get_Output_Matrix() const;
    std::pair<int,int>Get_Layer_Input_Information(int layer_number) const;
    Matrix& Get_Weights_Matrix(int layer_number);
    Matrix& Get_Bias_Matrix(int layer_number);

    Neural_Layer_Skeleton& Set_Layers(int layer_number);



private:
    // Private Data Members
    Matrix input_matrix;
    Matrix output_matrix;
    LossFunction lossFunction;
    float loss;
    std::vector<Neural_Layer_Skeleton> layers;
    bool input_matrix_constructed = false;
    bool output_matrix_constructed = false;
    bool loss_function_constructed = false;


    // Private Helper Methods
    void Compute_PreActivation_Matrix(Matrix &input_matrix_internal, Matrix &weights_matrix_internal,
                                      Matrix &bias_matrix_internal, Matrix &pre_activation_tensor_internal);
    void Compute_PostActivation_Matrix(Matrix &pre_activation_tensor_internal,
                                       Matrix &post_activation_tensor_internal, ActivationType activation_function_internal);
    void Apply_Activation(Matrix &pre_activation_tensor_internal, Matrix &post_activation_tensor_internal,
                          ActivationType activation_function);
    void Construct_Matrices();
    float ApplyActivationFunction(float value, ActivationType activation_type);

};



#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_BLOCKS_H
