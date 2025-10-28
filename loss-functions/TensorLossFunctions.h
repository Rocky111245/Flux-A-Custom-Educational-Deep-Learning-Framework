#ifndef TENSOR_LOSS_H
#define TENSOR_LOSS_H

#include "tensor-library/TensorLibrary.h"

// Loss choice
enum class tensor_loss_function {
    NONE,
    MSE_LOSS,                       // Mean Squared Error
    MAE_LOSS,                       // Mean Absolute Error
    HUBER_LOSS,                     // Huber loss (delta passed to Compute)
    CROSS_ENTROPY_LOSS,             // Binary Cross-Entropy (expects probabilities)
    CATEGORICAL_CROSS_ENTROPY_LOSS  // Categorical Cross-Entropy (expects per-row probabilities)
};

// Class that computes scalar loss and caches downstream error (∂L/∂a)
class Tensor_Loss_Function {
public:
    // Constructor. It basically asks what type of loss function to use and how many neurones are attached to this
    // specific loss head.

    explicit Tensor_Loss_Function(const tensor_loss_function type, const int neurone_count)
        : type_(type), loss_value_(0.0f),neurone_count_(neurone_count){}

    // Compute scalar loss and downstream error. For Huber, pass delta (>0).
    // huber_delta has default = 1.0f, ignored for other loss types.
    float Compute(const Tensor& predicted, const Tensor& target, float huber_delta = 1.0f);

    const Tensor& Get_Downstream_Error_View() const {return downstream_error_;}
    Tensor Get_Downstream_Error_Clone() const {return downstream_error_;}
    int Get_Neurone_Count() const {return neurone_count_;}
    tensor_loss_function Get_Type() const { return type_; }
    float Get_Loss() const { return loss_value_;}


private:
    tensor_loss_function type_;
    float        loss_value_;
    Tensor       downstream_error_;
    int          neurone_count_;


};

#endif // TENSOR_LOSS_H
float Tensor_Mean_Squared_Error(const Tensor& predicted, const Tensor& target);

float Tensor_Mean_Absolute_Error(const Tensor& predicted, const Tensor& target);

float Tensor_Huber_Loss(const Tensor& predicted, const Tensor& target, float delta);

float Tensor_Binary_Cross_Entropy(const Tensor& predicted, const Tensor& target) ;

float Tensor_Categorical_Cross_Entropy(const Tensor& predicted, const Tensor& target) ;


//APPENDIX:

// Output heads can have multiple neurons per head. Output heads are a bunch of neurons grouped together semantically
// and mechanically for a specific task.
// A single loss function typically operates on an entire output head. There can be multiple loss functions and multiple
// heads.
//
// A loss function typically expects a single output head (which can be a vector, matrix, or tensor). For example:
// Classification head: 1000 neurons → 1 categorical cross-entropy loss
// Regression head: 3 neurons (x,y,z coordinates) → 1 MSE loss
// Segmentation head: H×W×C tensor → 1 pixel-wise loss

//TODO: Add multiple loss function feature and align with neural block.cpp