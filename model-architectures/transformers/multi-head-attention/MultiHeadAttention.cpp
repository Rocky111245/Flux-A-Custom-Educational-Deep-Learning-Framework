//
// Created by rakib on 24/6/2025.
//

#include "MultiHeadAttention.h"


// Multi-head attention is basically a bunch of multiple single attention heads. This class HAS multiple single attention heads. This
// modular design gives us great flexibility in designing any number of attention blocks.


MultiHeadAttention::MultiHeadAttention(const Tensor& residual_stream_input, const int n_heads)
        : n_heads_(n_heads),    // Each head processes a fraction of the full embedding
          d_model_(residual_stream_input.columns()),
          d_k_(d_model_ / n_heads_),
          sequence_length_(residual_stream_input.rows()),
          batch_size_(residual_stream_input.depth()),
          residual_stream_input_copy_(residual_stream_input),
          concatenated_output_(sequence_length_, d_model_, batch_size_),
          W_o_(d_model_, d_model_, batch_size_),    //the final connecting weights that provide connectivity within single attention heads. Without this,
                                                    // the individual attention heads are not understood together to form a proper representation
          final_output_(sequence_length_, d_model_, batch_size_) // output after middle transformation
{
    // Validate that we can evenly divide the embedding dimension across heads
    if (n_heads_ <= 0) {
        throw std::invalid_argument("MultiHeadAttention::Constructor -> n_heads must be > 0");
    }
    if (d_model_ % n_heads_ != 0) {
        throw std::invalid_argument("MultiHeadAttention::Constructor -> d_model must be divisible by n_heads");
    }
    if (sequence_length_ <= 0 || batch_size_ <= 0 || d_model_ <= 0) {
        throw std::invalid_argument("MultiHeadAttention::Constructor -> Invalid tensor dimensions");
    }

    // Initialize all attention heads - each gets the same input but will learn different representations
    Initialize_All_Heads();

    // Initialize the output projection matrix that combines all head outputs
    Initialize_Output_Projection_Weights();
}

// Main interface - runs the complete multi-head attention computation
void MultiHeadAttention:: Forward_Pass() {
    Compute_All_Head_Outputs();         // Run each attention head independently
    Concatenate_Attention_Heads();     // Combine all head outputs into one tensor
    Apply_Output_Projection();        // Final linear transformation to d_model dimensions
}

// Getter for the final multi-head attention output.This is a copy. We prefer copies for now.
Tensor MultiHeadAttention::Get_Output_Clone() const {
    return final_output_;
}

Tensor& MultiHeadAttention::Get_Output_Mutable()  {
    return final_output_;
}

const Tensor& MultiHeadAttention::Get_Output_View() const {
    return final_output_;
}



// Step 1: Initialize all attention heads with appropriate input slices
void MultiHeadAttention::Initialize_All_Heads() {
    attention_heads_.clear();
    attention_heads_.reserve(n_heads_);
    // Pass the full input to the attention heads. They will divide the dimensions as needed.
    for (int head_idx = 0; head_idx < n_heads_; ++head_idx) {
        // This allows different heads to attend to different types of relationships
        attention_heads_.emplace_back(residual_stream_input_copy_, d_k_,true);
    }

    if (attention_heads_.size() != static_cast<size_t>(n_heads_)) {
        throw std::runtime_error("MultiHeadAttention::Initialize_All_Heads -> Head initialization count mismatch");
    }
}

// Step 2: Initialize the output projection matrix using Xavier initialization
void MultiHeadAttention::Initialize_Output_Projection_Weights() {
    assert(d_model_ > 0 && batch_size_ > 0 &&
        "MultiHeadAttention::Initialize_Output_Projection_Weights -> d_model or batch_size invalid");
    W_o_.Tensor_Xavier_Uniform_Share_Across_Depth();
}

// Step 3: Run forward pass through each attention head independently
void MultiHeadAttention::Compute_All_Head_Outputs() {
    // Each attention head computes its own query/key/value projections and attention weights
    // This parallelism allows the model to attend to different aspects simultaneously
    assert(!attention_heads_.empty() && "MultiHeadAttention::Compute_All_Head_Outputs -> No attention heads initialized");
    //This runs the heavy computation in all attention heads.
    for (int head_idx = 0; head_idx < n_heads_; ++head_idx) {
        attention_heads_[head_idx].Forward_Pass();
    }
}

// Step 4: Concatenate outputs from all attention heads
// Concatenate multiple tensors along the column dimension (for attention heads)
void MultiHeadAttention::Concatenate_Attention_Heads() {

    assert(static_cast<int>(attention_heads_.size()) == n_heads_ &&
           "Concatenate_Attention_Heads: attention_heads_.size() != n_heads_");
    assert(d_model_ == n_heads_ * d_k_ &&
           "Concatenate_Attention_Heads: d_model_ must equal n_heads_ * d_k_");
    assert(concatenated_output_.rows() == sequence_length_ &&
           concatenated_output_.columns() == d_model_ &&
           concatenated_output_.depth() == batch_size_ &&
           "Concatenate_Attention_Heads: concatenated_output_ shape mismatch");

    // For each batch and token position, copy each head's features into its slot

    assert(attention_heads_.size() == static_cast<size_t>(n_heads_) && "Attention head size should match n_heads_ "
                                                                       "in Concatenate_Attention_Heads");
    for (int d = 0; d < batch_size_; ++d) {
        for (int r = 0; r < sequence_length_; ++r) {
            int dest_col = 0;

            for (int h = 0; h < n_heads_; ++h) {

                const Tensor& out = attention_heads_[h].Get_Output_View(); // we only need a view

                // Shape checks per head
                assert(out.rows()  == sequence_length_ && "Head output rows != sequence_length_");
                assert(out.columns() == d_k_            && "Head output columns != d_k_");
                assert(out.depth() == batch_size_      && "Head output depth != batch_size_");

                // Copy the head's d_k_ features for (r, d)
                for (int c = 0; c < d_k_; ++c) {
                    concatenated_output_(r, dest_col + c, d) = out(r, c, d);
                }
                dest_col += d_k_;
            }

            assert(dest_col == d_model_ &&
                   "Concatenate_Attention_Heads: dest_col != d_model_ at row");
        }
    }
}

// Step 5: Apply final linear projection to transform concatenated features
void MultiHeadAttention::Apply_Output_Projection() {
    // Transform concatenated head outputs back to d_model dimensions
    // This allows the model to learn how to best combine information from different heads
    // Mathematical operation: final_output = concatenated_output * W_o
    assert(concatenated_output_.columns() == d_model_ && "MultiHeadAttention::Apply_Output_Projection -> Input columns must equal d_model");
    assert(W_o_.rows() == d_model_ && W_o_.columns() == d_model_ && W_o_.depth()==concatenated_output_.depth() &&
        "MultiHeadAttention::Apply_Output_Projection -> W_o shape mismatch");
    Tensor_Multiply_Tensor(final_output_, concatenated_output_, W_o_);
}

