//
// Created by Rocky170 on 7/22/2025.
//

#ifndef ACTIVATION_FUNCTION_DERIVATIVES_H
#define ACTIVATION_FUNCTION_DERIVATIVES_H

#include "Tensor_Library/Tensor_Library.h"
#include <cmath>

// Sigmoid derivatives
Tensor Sigmoid_Derivative(const Tensor &input);


// ReLU derivatives
Tensor Relu_Derivative(const Tensor &input);


// GELU derivatives
Tensor Gelu_Derivative(const Tensor &input);


// Leaky ReLU derivatives
Tensor Leaky_Relu_Derivative(const Tensor &input);


// Swish derivatives
Tensor Swish_Derivative(const Tensor &input);


// Tanh derivatives
Tensor Tanh_Derivative(const Tensor &input);


// Linear derivatives
Tensor Linear_Derivative(const Tensor &input);





#endif //ACTIVATION_FUNCTION_DERIVATIVES_H
