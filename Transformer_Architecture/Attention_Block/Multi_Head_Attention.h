//
// Created by rakib on 24/6/2025.
//
#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_MULTI_HEAD_ATTENTION_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_MULTI_HEAD_ATTENTION_H

#include "vector"
#include "Tensor_Library/Tensor_Library.h"
#include "Single_Attention_Head/Single_Attention_Head.h"

class Tensor;
class Single_Attention_Head;

class Multi_Head_Attention {
public:
    Multi_Head_Attention(const Tensor &batched_input, int n_heads);


        // Initialize all attention heads - each gets the same input but will learn different representations
        void Initialize_All_Heads();

        // Initialize the output projection matrix that combines all head outputs
        void Initialize_Output_Projection_Weights();


    // Main interface - runs the complete multi-head attention computation
    void Forward_Pass() ;
    // Getter for the final multi-head attention output.This is a copy. We prefer copies for now.
    Tensor Get_Output() const;

private:
    // Configuration parameters
    int n_heads_;                    // Number of parallel attention heads (typically 8, 12, or 16)
    int d_model_;                    // Original embedding dimension (e.g., 512, 768)
    int d_k_;                        // Dimension per head: d_model / n_heads
    int sequence_length_;            // Length of input sequences
    int batch_size_;                 // Number of sequences processed in parallel

    // Input tensor shared across all attention heads
    Tensor batched_input_;           // [sequence_length, d_model, batch_size]

    // The heart of multi-head attention: multiple parallel attention computations
    std::vector<Single_Attention_Head> attention_heads_;

    // Tensors for combining head outputs
    Tensor concatenated_output_;     // [sequence_length, d_model, batch_size] - all heads combined
    Tensor W_o_;                     // [d_model, d_model, batch_size] - output projection weights
    Tensor final_output_;            // [sequence_length, d_model, batch_size] - final result

    //Internal Functions
    void Compute_All_Head_Outputs();
    void Concatenate_Attention_Heads();
    void Apply_Output_Projection();
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_MULTI_HEAD_ATTENTION_H

