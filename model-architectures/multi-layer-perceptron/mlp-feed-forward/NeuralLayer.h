//
// Dense neural layer (fully connected + activation) with manual backprop.
//
// Overview
// --------
// A self-contained computational unit (layer) that maps an input tensor X ∈ R[S, I, B]
// (sequence length S, input features I, batch size B) to an output tensor
// A ∈ R[S, N, B] using learnable parameters:
//   - Weights W ∈ R[N, I, 1]
//   - Bias   b ∈ R[1, N, 1]

// Reuse
// -----
// The layer is designed to be composable. Higher-level "glue" classes can
// combine multiple instances to form complete models without modifying the
// internals. For example, an MLP block class can simply reuse the public
// API of this layer (Resize_Tensors, Initialize, Forward_Pass, Backpropagate,
// Update_Parameters) to implement a dense MLP stage.

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_NEURAL_LAYER_H



#include "tensor-library/TensorLibrary.h"

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


    explicit Neural(int neurons_in_layer, Activation_Type activation_type);
    void Forward_Pass();
    void Backpropagate(const Tensor& upstream_error_dL_da) ;


    void Resize_Tensors(const Tensor &input_tensor);
    void Initialize_Weights();
    void Initialize_Biases();
    void Pre_Activation_Matmul();
    void Apply_Activation_Function();
    void Compute_Local_Delta();
    void Compute_dL_db();
    void Compute_dL_dw();

    void Update_Parameters(float learning_rate);
    void Set_Upstream_Error(const Tensor& upstream_error_gradient);
    void Set_Input(const Tensor& input_tensor);

    const Tensor& Get_Downstream_Error();
    const Tensor&  Get_dL_dw() const;
    const Tensor&  Get_dL_db() const;
    const Tensor&  Get_Weights() const;
    const Tensor&  Get_Pre_Activation() const;
    const Tensor&  Get_Activation() const;
    const Tensor &Get_Input_View() const;

    Tensor Get_Input_Clone() const;

    const Tensor&  Get_Output_View() const;

    Tensor Get_Output_Clone() const;

    Activation_Type Get_Activation_Type() const;
    int Get_Neuron_Count() const;

    void Assert_Invariants() const;

    //Only call this after a full forward pass and backprop cycle
    void Clear_Local_Cache();


    //Mechanistic Interpretability related Functions. Only call after full model training. EXPERIMENTAL

    // Core interventions for post_activation_. neuron_idx is the neuron number in a particular layer
    void Ablate_Neuron(int neuron_idx);                           // Set to 0
    void Mean_Ablate_Neuron(int neuron_idx);                      // Replace with mean across batch/sequence
    void Patch_Activation(int neuron_idx, const Tensor& source);  // Replace with cached activation
    void Clamp_Neuron(int neuron_idx, float min_val, float max_val);     // Bound values
    void Scale_Neuron(int neuron_idx, float factor);              // Multiply by scalar
    void Randomize_Neuron(int neuron_idx, float min_val, float max_val); //Ramdomize

private:
    //Main variables
    Tensor input_;
    Tensor weight_;
    Tensor bias_;
    Tensor pre_activation_;
    Tensor post_activation_;
    Tensor output_;


    //Backpropagation related cache and inputs
    Tensor dL_da_upstream_layer_error_; //upstream error term received
    Tensor dL_dx_downstream_layer_error_; //error to be passed to downstream
    Tensor da_dz_; //f'(z) = derivative of the activation function
    Tensor dL_dz_; // local delta (dL/da ∘ da/dz) element-wise multiplication
    Tensor dL_dw_;
    Tensor dL_db_;


    int number_of_neurons_;         // Total number of neurones. This is basically a feature expansion or
    //contraction depending on the neurone number. If neuron number>features in dataset,it is a feature expansion. Representation of a vector into
    //higher dimensions.
    Activation_Type activation_type_;


};


//Let's derive the chain rule for the backpropagation.
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




