
// Created by rakib on 25/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LAYER_NORMALIZATION_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LAYER_NORMALIZATION_H

#include "Tensor_Library/Tensor_Library.h"
#include <vector>
#include <cmath>

class Layer_Norm {
public:
    explicit Layer_Norm(const int d_model, float epsilon = 1e-5f, float gamma_parameter=1.0f, float beta_parameter=0.0f);

    //this basically does the operation on the residual stream directly,it modifies the residual stream
    void Apply_Layer_Norm(Tensor& residual_stream);
    void Apply_RMS_Norm(Tensor& residual_stream, bool use_beta=false);

    std::vector<float>& Get_Gamma();
    std::vector<float>& Get_Beta();
    const std::vector<float>& Get_Gamma() const;
    const std::vector<float>& Get_Beta() const;



private:
    // Model parameters
    float epsilon_;                  // Small constant for numerical stability
    float initial_gamma_;            //IF we need this for some reason
    float initial_beta_;             //IF we need this for some reason
    std::vector<float> gamma_;       // Learnable scale parameters [d_model]
    std::vector<float> beta_;        // Learnable shift parameters [d_model]

    float Compute_Mean(const std::vector<float>& row) const;

    float Compute_Mean_Square(const std::vector<float>& row) const;

    float Compute_Variance(const std::vector<float>& row, float mean) const;
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_LAYER_NORMALIZATION_H