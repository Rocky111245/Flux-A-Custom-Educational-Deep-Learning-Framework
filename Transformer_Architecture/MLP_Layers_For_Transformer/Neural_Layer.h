//
// Created by rakib on 09/07/2025.
// The methodology is that, this is a self containing layer.No original data being passed through this gets transformed.Every data passed through this has its own copy internally
//via private variables.

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_H



#include "Tensor_Library/Tensor_Library.h"

// Activation function enumeration
enum class Activation_Type {
    RELU,
    GELU,
    LEAKY_RELU,
    SWISH,
    TANH,
    SIGMOID,
    LINEAR
};

class Neural {
public:


    explicit Neural(const Tensor& input_tensor,int neurons_in_layer, Activation_Type activation_type);
    void Forward_Pass();
    void Backpropagate(const Tensor& upstream_error_dL_da) ;


    const Tensor& Get_Downstream_Error();
    const Tensor&  Get_dL_dw() const;
    const Tensor&  Get_dL_db() const;
    const Tensor&  Get_Weights() const;
    const Tensor&  Get_Pre_Activation() const;
    const Tensor&  Get_Activation() const;
    const Tensor&  Get_Output() const;
    int Get_Neuron_Count() const;



    void Update_Parameters(float learning_rate);
    void Set_Upstream_Error(const Tensor& upstream_error_gradient);
    void Set_Input(const Tensor& input_tensor);

    //Only call this after a full forward pass and backprop cycle
    void Clear_Local_Cache();





private:
    //Main variables
    Tensor input_;
    Tensor weight_;
    Tensor bias_;
    Tensor pre_activation_;
    Tensor post_activation_;
    Tensor output_;


    //Backpropagation related cache and inputs
    Tensor dL_da_upstream_layer_error; //upstream error term received
    Tensor dL_dx_downstream_layer_error; //error to be passed to downstream
    Tensor da_dz_; //f'(z) = derivative of the activation function
    Tensor dL_dz_; // local delta (dL/da ∘ da/dz) element-wise multiplication
    Tensor dL_dw_;
    Tensor dL_db_;
    

    int number_of_neurons_;         // Total number of neurones we want to data to pass through. This is basically a feature expansion or
    //contraction depending on the neurone number. If neuron number>features in dataset,it is a feature expansion. Representation of a vector into
    //higher dimensions.
    Activation_Type activation_type_;



    void Initialize_Weights_And_Biases();


    void Apply_Activation_Function();




    void Pre_Activation_Matmul();

    void Compute_Local_Delta();

    void Compute_dL_db();
    void Compute_dL_dw();

};


//Let us derive the chain rule for the backpropagation.
// --- Chain Rule for Backpropagation ---
//
// The chain rule allows computation of gradients for weight (w) and bias (b) updates by breaking down dependencies:
// 1. z = w * a_in + b (pre-activation)
// 2. A = f(z) (post-activation)
// 3. L depends on A (loss function)
//
// Gradients are computed as:
// - dL/dw = (dL/dA) * (dA/dz) * (dz/dw) = upstream gradient * activation derivative * input
// - dL/db = (dL/dA) * (dA/dz) * (dz/db) = upstream gradient * activation derivative (since dz/db = 1)
// - dL/da_in = (dL/dA) * (dA/dz) * (dz/da_in) = upstream gradient * activation derivative * weight (for upstream propagation)
//
// Execution order during backpropagation:
// 1. Set upstream error (dL/dA)
// 2. Compute local delta (dL/dA * dA/dz)
// 3. Compute weight gradients (dL/dw)
// 4. Compute bias gradients (dL/db)
// 5. Compute upstream error (dL/da_in)
//
// This implementation efficiently computes gradients for parameter updates using the chain rule.
// Note: This is not autograd but purely manual backpropagation.


#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_H




