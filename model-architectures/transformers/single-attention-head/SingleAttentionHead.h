//
// Created by rakib on 23/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_SINGLE_ATTENTION_HEAD_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_SINGLE_ATTENTION_HEAD_H

#include <limits>
#include <cmath>
#include "tensor-library/TensorLibrary.h"

// Forward declarations
class Tensor;
class Matrix;

// Single attention head implementing scaled dot-product attention mechanism
// Input format: [sequence_length, d_model, batch_size]
// Output format: [sequence_length, d_k, batch_size]
class SingleAttentionHead {
public:
    // Constructor takes input tensor and head dimension
    // d_k is typically d_model divided by number of heads
    SingleAttentionHead(const Tensor &residual_stream_input, int d_k, bool masked_attention);



    // Main function - runs the complete attention computation
    void Forward_Pass();
    Tensor Get_Output_Clone() const;
    const Tensor &Get_Output_View() const;
    Tensor &Get_Output_Mutable();

    // Returns the computed attention output tensor
private:
    // Basic configuration
    int d_k_;                        // Head projection size
    int sequence_length_;            // Length of input sequences
    int d_model_;                    // Original embedding dimension
    int batch_size_;                 // Number of sequences processed together

    // Input binary-serializer
    Tensor residual_stream_input_copy_;           // Input tensor [sequence_length, d_model, batch_size]

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

    bool masked_attention_;          // Checks if we want masked attention. For now,
                                    //I didn't add the pad masking since during training we can drop the last batch(for now).

    // Step-by-step computation functions
    void Initialize_Weights();          // Set up Q, K, V weight matrices using Xavier initialization
    void Compute_Projections();         // Transform input into Q, K, V representations
    void Compute_Attention_Scores();    // Calculate similarity scores between queries and keys
    void Apply_Softmax_To_Attention_Scores();  // Convert scores to probability distributions
    void Compute_Attention_Output();    // Generate final context-aware representations
    void Apply_Causal_Mask();
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_SINGLE_ATTENTION_HEAD_H