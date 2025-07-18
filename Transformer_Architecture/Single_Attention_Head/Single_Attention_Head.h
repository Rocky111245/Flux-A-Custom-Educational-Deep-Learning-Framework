//
// Created by rakib on 23/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_SINGLE_ATTENTION_HEAD_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_SINGLE_ATTENTION_HEAD_H

#include <limits>
#include <cmath>
#include "Tensor_Library/Tensor_Library.h"
#include "MatrixLibrary.h"

// Forward declarations
class Tensor;
class Matrix;

// Single attention head implementing scaled dot-product attention mechanism
// Input format: [sequence_length, d_model, batch_size]
// Output format: [sequence_length, d_k, batch_size]
class Single_Attention_Head {
public:
    // Constructor takes input tensor and head dimension
    // d_k is typically d_model divided by number of heads
    Single_Attention_Head(const Tensor &batched_input, int d_k);



    // Main function - runs the complete attention computation
    void Forward_Pass();

    // Returns the computed attention output tensor
    Tensor Get_Output() const;

private:
    // Basic configuration
    int d_k_;                        // Head projection size
    int sequence_length_;            // Length of input sequences
    int d_model_;                    // Original embedding dimension
    int batch_size_;                 // Number of sequences processed together

    // Input data
    Tensor batched_input_;           // Input tensor [sequence_length, d_model, batch_size]

    // Learnable weight matrices - these get trained
    Tensor W_q_;                     // Query projection weights [d_model, d_k, batch_size]
    Tensor W_k_;                     // Key projection weights [d_model, d_k, batch_size]
    Tensor W_v_;                     // Value projection weights [d_model, d_k, batch_size]

    // Intermediate computation results
    Tensor Q_;                       // Query projections [sequence_length, d_k, batch_size]
    Tensor K_;                       // Key projections [sequence_length, d_k, batch_size]
    Tensor V_;                       // Value projections [sequence_length, d_k, batch_size]
    Tensor attention_scores_;        // Raw attention scores [sequence_length, sequence_length, batch_size]
    Tensor attention_weights_;       // Normalized attention weights [sequence_length, sequence_length, batch_size]
    Tensor output_;                  // Final attention output [sequence_length, d_k, batch_size]

    // Step-by-step computation functions
    void Initialize_Weights();          // Set up Q, K, V weight matrices using Xavier initialization
    void Compute_Projections();         // Transform input into Q, K, V representations
    void Compute_Attention_Scores();    // Calculate similarity scores between queries and keys
    void Apply_Softmax_To_Attention_Scores();  // Convert scores to probability distributions
    void Compute_Attention_Output();    // Generate final context-aware representations
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_SINGLE_ATTENTION_HEAD_H