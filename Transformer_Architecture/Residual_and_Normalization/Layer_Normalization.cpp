//
// Created by rakib on 25/6/2025.
//

#include "Tensor_Library/Tensor_Library.h"
#include <vector>
#include <cmath>
#include "Layer_Normalization.h"

// Constructor: Initialize with feature dimension size
 Layer_Norm::Layer_Norm(const int d_model, const float epsilon, const float gamma_parameter, const float beta_parameter)
    : epsilon_(epsilon),
      initial_gamma_(gamma_parameter),
      initial_beta_(beta_parameter),
      gamma_(d_model, gamma_parameter),    // Initialize gamma to 1.0 (no scaling initially)
      beta_(d_model, beta_parameter) {    // Initialize beta to 0.0 (no shift initially)
}

// Apply_Layer_Norm pass: Apply layer normalization to residual_stream tensor. Input tensor is the residual stream
void Layer_Norm::Apply_Layer_Norm(Tensor& residual_stream) {
     const int d_model = residual_stream.columns();
     if (residual_stream.rows() <= 0 || d_model <= 0 || residual_stream.depth() <= 0) {
         throw std::invalid_argument("Invalid residual stream dimensions in Layer_Norm::Apply_Layer_Norm");
     }
     if (gamma_.size() != d_model || beta_.size() != d_model) {
         throw std::logic_error("Layer_Norm::Apply_Layer_Norm: gamma/beta size mismatch with d_model");
     }

    // Apply layer normalization for each (sequence_pos, batch) pair
     for (int seq = 0; seq < residual_stream.rows(); ++seq) {
         for (int batch = 0; batch < residual_stream.depth(); ++batch) {
             const std::vector<float> row = residual_stream.Get_Row_Vector(seq, batch);
             float mean = Compute_Mean(row);
             float var = Compute_Variance(row, mean);
             float inv_std = 1.0f / std::sqrt(var + epsilon_);


             for (int d = 0; d < d_model; ++d) {
                 float norm = (row[d] - mean) * inv_std;
                 residual_stream(seq, d, batch) = gamma_[d] * norm + beta_[d];
             }
         }
     }
}

// RMS Norm Implementation
void Layer_Norm::Apply_RMS_Norm(Tensor& residual_stream, bool use_beta) {
     const int d_model = residual_stream.columns();
     if (residual_stream.rows() <= 0 || d_model <= 0 || residual_stream.depth() <= 0) {
         throw std::invalid_argument("Layer_Norm::Apply_RMS_Norm: invalid residual shape");
     }
     if (gamma_.size() != d_model) {
         throw std::logic_error("Layer_Norm::Apply_RMS_Norm: gamma size mismatch with d_model");
     }


     for (int seq = 0; seq < residual_stream.rows(); ++seq) {
         for (int batch = 0; batch < residual_stream.depth(); ++batch) {
             const std::vector<float> row = residual_stream.Get_Row_Vector(seq, batch);
             float rms = std::sqrt(Compute_Mean_Square(row) + epsilon_);
             float inv_rms = 1.0f / rms;


             for (int d = 0; d < d_model; ++d) {
                 float norm = gamma_[d] * (row[d] * inv_rms);
                 residual_stream(seq, d, batch) = use_beta ? (norm + beta_[d]) : norm;
             }
         }
     }
 }

// Getters for learnable parameters (useful for training/optimization)
std::vector<float>& Layer_Norm::Get_Gamma()
{ return gamma_; }

std::vector<float>& Layer_Norm::Get_Beta()
{ return beta_; }


//read only
const std::vector<float>& Layer_Norm::Get_Gamma() const
{ return gamma_; }

const std::vector<float>&Layer_Norm:: Get_Beta()const
{ return beta_; }


//PRIVATE FUNCTIONS (INTERNAL USE) :

// Helper function: Compute mean across d_model dimension
float Layer_Norm::Compute_Mean(const std::vector<float>& row) const {
     float sum = 0.0f;
     for (float v : row) sum += v;
     return sum / static_cast<float>(row.size());
 }

// Helper function: Compute variance across d_model dimension
float Layer_Norm::Compute_Variance(const std::vector<float>& row, float mean) const {
     float sum_sq_diff = 0.0f;
     for (float v : row) {
         float diff = v - mean;
         sum_sq_diff += diff * diff;
     }
     return sum_sq_diff / static_cast<float>(row.size());
 }

// Helper function: Compute mean square
float Layer_Norm::Compute_Mean_Square(const std::vector<float>& row) const {
     float sum_sq = 0.0f;
     for (float v : row) sum_sq += v * v;
     return sum_sq / static_cast<float>(row.size());
 }

