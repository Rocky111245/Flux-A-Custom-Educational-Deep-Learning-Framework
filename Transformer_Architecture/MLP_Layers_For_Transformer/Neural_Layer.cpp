//
// Created by rakib on 09/07/2025.
//

//Updated on 08/23/2025

#include "Neural_Layer.h"
#include "MatrixLibrary.h"
#include "Tensor_Library/Tensor_Library.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <Activation_Function_Derivatives/Activation_Function_Derivatives.h>
#include <Activation_Functions/Activation_Functions.h>



// Constructor. The input passed to an MLP layer is copied internally. The input tensor for NLP purposes is
// [seq_length,d_model,batch_size]
Neural::Neural(const Tensor& input_tensor, const int neurons_in_layer, const Activation_Type activation_type)
    : input_(input_tensor)
    , number_of_neurons_(neurons_in_layer)
    // Weight tensor: [ neurons,input_features, 1] - shared across all batches
    , weight_(neurons_in_layer,input_tensor.columns(), 1)
    // Bias tensor: [1, neurons, 1] - shared across all batches
    , bias_(1, neurons_in_layer, 1)
    // Pre-activation tensor: [sequence_length, neurons, batch_size]
    , pre_activation_(input_tensor.rows(), neurons_in_layer, input_tensor.depth())
    // Activation tensor: [sequence_length, neurons, batch_size]
    , post_activation_(input_tensor.rows(), neurons_in_layer, input_tensor.depth())
    // Output tensor: [sequence_length, neurons, batch_size]
    , output_(input_tensor.rows(), neurons_in_layer, input_tensor.depth())
    , da_dz_(input_tensor.rows(), neurons_in_layer, input_tensor.depth())
    , dL_dw_(neurons_in_layer,input_tensor.columns(), 1)
    , dL_db_(1, neurons_in_layer, 1)
    , dL_dz_(input_tensor.rows(), neurons_in_layer, input_tensor.depth()) //same dimensions as pre-activation
    , dL_dx_downstream_layer_error(input_tensor.rows(),input_tensor.columns(),input_tensor.depth())
    ,activation_type_(activation_type)
    , dL_da_upstream_layer_error(input_tensor.rows(), number_of_neurons_, input_tensor.depth())

{
    if (neurons_in_layer <= 0) { throw std::invalid_argument("neurons_in_layer must be > 0"); }

    assert(input_.rows()    > 0 && input_.columns() > 0 && input_.depth() > 0 && "Input tensor must have positive "
                                                                                 "dimensions in Neural Constructor");

    // Initialize weights and biases
    Initialize_Weights_And_Biases();

    // After init, confirm key shapes
    assert(weight_.rows()    == number_of_neurons_ &&
           weight_.columns() == input_.columns()   &&
           weight_.depth()   == 1 &&
           "Weight tensor must be [N, I, 1] in Neural Constructor");

    assert(bias_.rows() == 1 && bias_.columns() == number_of_neurons_ && bias_.depth() == 1 &&
           "Bias tensor must be [1, N, 1] in Neural Constructor");

    assert(pre_activation_.rows() == input_.rows() &&
           pre_activation_.columns() == number_of_neurons_ &&
           pre_activation_.depth() == input_.depth() &&
           "pre_activation_ must be [S, N, B] in Neural Constructor");

    assert(post_activation_.rows() == pre_activation_.rows() &&
           post_activation_.columns() == pre_activation_.columns() &&
           post_activation_.depth() == pre_activation_.depth() &&
           "post_activation_ must match pre_activation_ shape in Neural Constructor");

}

// Pre-activation matrix multiplication
void Neural::Pre_Activation_Matmul() {
    // Perform tensor-matrix multiplication for each batch slice
    // Input: [sequence_length, input_features, batch_size]
    // Weight: [neurons,input_features, 1]`
    // Output: [sequence_length, neurons, batch_size]

    //Step 1: First transpose the weights
    // Before transpose
    assert(weight_.rows()==number_of_neurons_ && weight_.columns()==input_.columns() && weight_.depth()==1 &&
           "Weight must be [N,I,1] in Pre_Activation_Matmul");
    assert(bias_.rows()==1 && bias_.columns()==number_of_neurons_ && bias_.depth()==1 &&
           "Bias must be [1,N,1] in Pre_Activation_Matmul");

    Tensor transposed_broadcasted_weight_tensor(weight_.columns(),weight_.rows(),weight_.depth());
    Tensor_Transpose(transposed_broadcasted_weight_tensor,weight_);
    assert(transposed_broadcasted_weight_tensor.rows()==input_.columns() && transposed_broadcasted_weight_tensor.columns()
        == number_of_neurons_ && " The transposition in Pre_Activation_Matmul has a dimension bug ");

    //Step 2: Broadcast the weights
    Tensor x;
    Tensor_Broadcast_At_Depth(x,transposed_broadcasted_weight_tensor,input_.depth());
    assert(x.depth() == input_.depth() && "Broadcasted weight depth mismatch with input in Pre_Activation_Matmul");
    assert(x.rows() == input_.columns() && x.columns() == number_of_neurons_ && "Broadcasted weight dimensions are incorrect"
                                                                                "in Pre_Activation_Matmul");

    transposed_broadcasted_weight_tensor=std::move(x);// we dont need x anymore,so we destroy it by moving

    //Step 2: Do the multiplication between the weights and the inputs.
    //Input has shape [seq_length,d_model,batch_size], transposed weight matrix has shape [d_model,neurones,batch size]
    Tensor_Multiply_Tensor(pre_activation_,transposed_broadcasted_weight_tensor,input_);
    assert(pre_activation_.rows() == input_.rows() &&
           pre_activation_.columns() == number_of_neurons_ &&
           pre_activation_.depth() == input_.depth() &&
           "Pre-activation dimensions mismatch after matmul in Pre_Activation_Matmul");

    //Step 3: Broadcast the Bias and apply it. Apply bias to all sequence positions and batch elements
    for (int depth = 0; depth < pre_activation_.depth(); ++depth) {
        for (int rows = 0; rows < pre_activation_.rows(); ++rows) {
            for (int column = 0; column < pre_activation_.columns(); ++column) {
                pre_activation_(rows, column, depth) += bias_(0, column, 0);
            }
        }
    }
}


// Private helper methods
void Neural::Initialize_Weights_And_Biases() {
    weight_.Tensor_Xavier_Uniform_MLP(input_.columns(), number_of_neurons_);
    // Bias to zeros:
    dL_db_.Fill(0.0f);
    for (int c = 0; c < bias_.columns(); ++c) bias_(0, c, 0) = 0.0f;
}



void Neural::Apply_Activation_Function() {
    // Apply activation function based on the selected type
    assert(output_.rows() == post_activation_.rows() &&
       output_.columns() == post_activation_.columns() &&
       output_.depth() == post_activation_.depth() &&
       "Output buffer must match activation shape Apply_Activation_Function early check");

    assert(pre_activation_.rows() == post_activation_.rows() &&
       pre_activation_.columns() == post_activation_.columns() &&
       pre_activation_.depth() == post_activation_.depth() &&
       "Activation function: dst and src must have the same shape in Apply_Activation_Function inside MLP");

    switch (activation_type_) {
        case Activation_Type::RELU:
            ReLU(post_activation_, pre_activation_);
            break;

        case Activation_Type::GELU:
            GELU(post_activation_, pre_activation_);
            break;

        case Activation_Type::LEAKY_RELU:
            Leaky_ReLU(post_activation_, pre_activation_);
            break;

        case Activation_Type::SWISH:
            Swish(post_activation_, pre_activation_);
            break;

        case Activation_Type::TANH:
            Tanh(post_activation_, pre_activation_);
            break;

        case Activation_Type::SIGMOID:
            Sigmoid(post_activation_, pre_activation_);
            break;

        case Activation_Type::LINEAR:
            Linear(post_activation_, pre_activation_);
            break;

        default:
            throw std::invalid_argument("Unknown activation function type");
    }
}



// Complete forward pass
void Neural::Forward_Pass() {
    // Step 1: Linear transformation
    Pre_Activation_Matmul();

    // Step 2: Apply activation function
    Apply_Activation_Function();

    assert(weight_.columns() == input_.columns() && "Weight/input feature mismatch in Forward_Pass");
    assert(bias_.columns() == number_of_neurons_ && "Bias/out_features mismatch in Forward_Pass");

    // Step 3: Copy to output tensor
    output_ = post_activation_;
}


// This section is for manual backpropagation. Error tensor MUST be in the shape of the input of the upstream layer.
void Neural::Backpropagate(const Tensor& upstream_error_dL_da) {

    //Step 1: Set the Upstream Error first, without this,we cannot compute backprop
    Set_Upstream_Error(upstream_error_dL_da);
    //Step 2: First compute the local delta (dL/dz)
    Compute_Local_Delta();

    //Step 3: Then find the dL_dw
    Compute_dL_dw();

    //Step 4: Then find the dL_db
    Compute_dL_db();

}

//Step 1 in Backpropagation. The upstream error gradient flowing should have the exact same shape as this layer's output.
void Neural::Set_Upstream_Error(const Tensor& upstream_error_gradient) {
    assert(output_.rows()>0 && output_.columns()>0 && output_.depth()>0 &&"Output must be allocated before "
                                                                          "setting upstream error in Set_Upstream_Error");
    if (upstream_error_gradient.rows() != output_.rows() ||
        upstream_error_gradient.columns() != output_.columns() ||
        upstream_error_gradient.depth() != output_.depth()) {throw std::invalid_argument("Upstream gradient shape mismatch in Set_Upstream_Error");}

    assert(dL_da_upstream_layer_error.rows()==upstream_error_gradient.rows() && dL_da_upstream_layer_error.columns() == upstream_error_gradient.columns()
        && dL_da_upstream_layer_error.depth() == upstream_error_gradient.depth() && "Dimension mismatch for error propagation in"
                                                                 "Set_Upstream_Error in MLP");
    dL_da_upstream_layer_error = upstream_error_gradient;
}



//// Computes the local gradient (delta) of the current layer for backpropagation.
//
// This function performs two key operations:
// 1. Computes the derivative of the activation function (∂a/∂z) based on the selected activation type,
//    using the pre-activation tensor `pre_activation_`.
// 2. Computes the element-wise product between the upstream gradient `dL_da_upstream_layer`
//    (∂L/∂a from the next layer) and the local activation derivative (∂a/∂z) to obtain ∂L/∂z,
//    which is stored in `dL_dz_` for use in parameter gradient computation and propagating error.
//
// Mathematically: dL_dz_ = dL/da ⊙ da/dz (elementwise operation)

void Neural::Compute_Local_Delta() {
    // Step 2: Compute da/dz for the activation function depending on activation type
    assert(da_dz_.rows()==pre_activation_.rows() && da_dz_.columns() == pre_activation_.columns() && da_dz_.depth() == pre_activation_.depth()
        && "Dimension mismatch in Compute_Local_Delta inside MLP");

    assert(dL_da_upstream_layer_error.rows()==post_activation_.rows() &&
           dL_da_upstream_layer_error.columns()==post_activation_.columns() &&
           dL_da_upstream_layer_error.depth()==post_activation_.depth() &&
           "Upstream dL/da must match this layer's activation shape");

    assert(dL_dz_.rows()==pre_activation_.rows() &&
           dL_dz_.columns()==pre_activation_.columns() &&
           dL_dz_.depth()==pre_activation_.depth() &&
           "dL_dz_ buffer must be [S,N,B]");


    switch (activation_type_) {
        case Activation_Type::SIGMOID:
            da_dz_ = Sigmoid_Derivative(pre_activation_);
            break;
        case Activation_Type::RELU:
            da_dz_ = Relu_Derivative(pre_activation_);
            break;
        case Activation_Type::GELU:
            da_dz_ = Gelu_Derivative(pre_activation_);
            break;
        case Activation_Type::LEAKY_RELU:
            da_dz_ = Leaky_Relu_Derivative(pre_activation_);
            break;
        case Activation_Type::SWISH:
            da_dz_ = Swish_Derivative(pre_activation_);
            break;
        case Activation_Type::TANH:
            da_dz_ = Tanh_Derivative(pre_activation_);
            break;
        case Activation_Type::LINEAR:
            da_dz_ = Linear_Derivative(pre_activation_);
            break;
        default:
            throw std::runtime_error("Unknown activation type in Get_dL_dw()");
    }

    // Step 2: Compute dL/dz
    assert(dL_dz_.rows()==dL_da_upstream_layer_error.rows() && dL_dz_.columns() == dL_da_upstream_layer_error.columns() &&
        dL_dz_.depth() == dL_da_upstream_layer_error.depth() && "Dimension mismatch in Compute_Local_Delta()");
    Tensor_Multiply_Tensor_ElementWise(dL_dz_, dL_da_upstream_layer_error, da_dz_);

}


//Needs local delta (dL/dz) to be computed first.
//dl/dW= dL/dz * dz/dW (which is basically the input)
void Neural::Compute_dL_dw() {
    // dL/dz must align with input along S and B
    assert(dL_dz_.rows()==input_.rows() && dL_dz_.depth()==input_.depth() &&
           "dL_dz and input must share sequence length and batch in Compute_dL_dw");
    assert(dL_dw_.rows()==number_of_neurons_ && dL_dw_.columns()==input_.columns() && dL_dw_.depth()==1 &&
           "dL_dw_ must be [N,I,1]");
    dL_dw_.Fill(0.0f);

    // dL_dz_ has shape [input_tensor.rows(), neurons_in_layer, input_tensor.depth()]
    // dz/dW has shape  [input_tensor.rows(),input_tensor.cols(),input_tensor.depth()]
    // transposed dz/dW (input) has shape [input_tensor.cols(),input_tensor.rows(),input_tensor.depth()]

    Tensor input_transposed(input_.columns(), input_.rows(), input_.depth());
    Tensor_Transpose(input_transposed, input_);

    // Temporary output: [I, N, B]
    Tensor temp_dL_dw(input_transposed.rows(), dL_dz_.columns(), input_transposed.depth());
    Tensor_Multiply_Tensor(temp_dL_dw, input_transposed, dL_dz_);

    assert(temp_dL_dw.rows()==input_.columns() && temp_dL_dw.columns()==dL_dz_.columns() &&
              temp_dL_dw.depth()==input_.depth() && "temp_dL_dw must be [I,N,B]");

    // Reduce over sequence and batch: temp [I, N, B] → dL_dw_ [N, I, 1]
    for (int b = 0; b < temp_dL_dw.depth(); ++b) {
        for (int i = 0; i < temp_dL_dw.rows(); ++i) {
            for (int n = 0; n < temp_dL_dw.columns(); ++n) {
                dL_dw_(n, i, 0) += temp_dL_dw(i, n, b);
            }
        }
    }
}

//Needs local delta (dL/dz) to be computed first
//dl/db= dL/dz * dz/db
void Neural::Compute_dL_db() {

    assert(dL_db_.rows()==1 && dL_db_.columns()==number_of_neurons_ && dL_db_.depth()==1 &&
           "dL_db_ must be [1,N,1]");
    assert(dL_dz_.columns()==number_of_neurons_ && "dL_dz_ must have N columns to accumulate bias gradients");

    dL_db_.Fill(0.0f);
    // Accumulate gradients
    for (int d = 0; d < dL_dz_.depth(); ++d) {
        for (int r = 0; r < dL_dz_.rows(); ++r) {
            for (int c = 0; c < dL_dz_.columns(); ++c) {
                dL_db_(0, c, 0) += dL_dz_(r, c, d);
            }
        }
    }
}

//Only call this after a full forward pass and backprop cycle otherwise it will wipe out important gradients.
void Neural::Clear_Local_Cache() {
    // Clear gradients
    dL_dw_.Fill(0.0f);
    dL_db_.Fill(0.0f);
    dL_dz_.Fill(0.0f);
    dL_da_upstream_layer_error.Fill(0.0f);
    dL_dx_downstream_layer_error.Fill(0.0f);
    da_dz_.Fill(0.0f);
    pre_activation_.Fill(0.0f);
    post_activation_.Fill(0.0f);
    output_.Fill(0.0f);
}



// This method is used to retrieve the error term dL_da to be passed to the downstream layer.Needs local delta (dL/dz) to be computed first.
// Requires: Backpropagate(...) or Compute_Local_Delta() already called.
// Returns dL/dx with shape [S, I, B].
const Tensor& Neural::Get_Downstream_Error() {
    assert(dL_dz_.rows()==input_.rows() && dL_dz_.columns()==number_of_neurons_ && dL_dz_.depth()==input_.depth() &&
           "dL_dz_ must be [S,N,B] in Get_Downstream_Error");

    assert(weight_.depth()==1 && "Weight depth must be 1 before broadcast in Get_Downstream_Error");

    if (dL_dx_downstream_layer_error.rows()    != input_.rows()   ||
        dL_dx_downstream_layer_error.columns() != input_.columns()||
        dL_dx_downstream_layer_error.depth()   != input_.depth()) {
        dL_dx_downstream_layer_error = Tensor(input_.rows(), input_.columns(), input_.depth());
        }

    assert(dL_dx_downstream_layer_error.rows()==input_.rows() &&
           dL_dx_downstream_layer_error.columns()==input_.columns() &&
           dL_dx_downstream_layer_error.depth()==input_.depth() &&
           "Downstream error buffer must be [S,I,B] in Get_Downstream_Error");

    // Broadcast weights to [N,I,B]
    Tensor W_broadcast;
    Tensor_Broadcast_At_Depth(W_broadcast, weight_, input_.depth());
    assert(W_broadcast.rows()==weight_.rows() && W_broadcast.columns()==weight_.columns() &&
           W_broadcast.depth()==input_.depth() && "Broadcasted W must be [N,I,B] in Get_Downstream_Error");

    // [S,N,B] x [N,I,B] → [S,I,B]
    Tensor_Multiply_Tensor(dL_dx_downstream_layer_error, dL_dz_, W_broadcast);
    return dL_dx_downstream_layer_error;
}






void Neural::Update_Parameters(float learning_rate) {

    if (!std::isfinite(learning_rate) || learning_rate <= 0.0f) {
        throw std::invalid_argument("learning_rate must be finite and > 0");
    }

    assert(dL_dw_.rows()==weight_.rows() && dL_dw_.columns()==weight_.columns() && dL_dw_.depth()==1 &&
          "dL_dw_ shape must match weight shape");
    assert(dL_db_.rows()==1 && dL_db_.columns()==bias_.columns() && dL_db_.depth()==1 &&
           "dL_db_ shape must match bias shape");

    for (int r = 0; r < weight_.rows(); ++r) {
        for (int c = 0; c < weight_.columns(); ++c) {
            weight_(r, c, 0) -= learning_rate * dL_dw_(r, c, 0);
        }
    }

    for (int c = 0; c < bias_.columns(); ++c) {
        bias_(0, c, 0) -= learning_rate * dL_db_(0, c, 0);
    }

}

// Getter methods

//Needs local delta (dL/dz) to be computed first. Call Backprop function first before using.
const Tensor&  Neural::Get_dL_dw() const{
    return dL_dw_;
}

//Needs local delta (dL/dz) to be computed first. Call Backprop function first before using.
const Tensor& Neural::Get_dL_db() const {
    return dL_db_;
}

const Tensor& Neural::Get_Output() const {
    return output_;
}

const Tensor& Neural::Get_Weights() const {
    return weight_;
}

const Tensor&  Neural::Get_Pre_Activation() const {
    return pre_activation_;
}

const Tensor&  Neural::Get_Activation() const {
    return post_activation_;
}

int Neural::Get_Neuron_Count() const {
    return number_of_neurons_;
};



void Neural::Set_Input(const Tensor& input_tensor) {
    if (input_tensor.rows()    != input_.rows() ||
        input_tensor.columns() != input_.columns() ||
        input_tensor.depth()   != input_.depth()) {
        throw std::invalid_argument("Input shape mismatch");

        }
    assert(input_tensor.rows()>0 && input_tensor.columns()>0 && input_tensor.depth()>0 &&
       "Input must have positive dimensions in Set_Input");

    input_ = input_tensor;
}



