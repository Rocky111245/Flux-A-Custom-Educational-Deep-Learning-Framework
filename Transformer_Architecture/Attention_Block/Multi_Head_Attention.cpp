//
// Created by rakib on 24/6/2025.
//

#include "Multi_Head_Attention.h"


// Multi-head attention is basically a bunch of multiple single attention heads. This class HAS multiple single attention heads. This
// modular design gives us great flexibility in designing any number of attention blocks.


Multi_Head_Attention::Multi_Head_Attention(const Tensor& batched_input, int n_heads)
        : batched_input_(batched_input),    //  input goes in
          n_heads_(n_heads),
          sequence_length_(batched_input.rows()),
          d_model_(batched_input.columns()),
          batch_size_(batched_input.depth()),
          d_k_(d_model_ / n_heads_),  // Each head processes a fraction of the full embedding
          concatenated_output_(sequence_length_, d_model_, batch_size_),
          W_o_(d_model_, d_model_, batch_size_),    //the final connecting weights that provide connectivity within single attention heads. Without this,
                                                    // the individual attention heads are not understood together to form a proper representation
          final_output_(sequence_length_, d_model_, batch_size_) // output after middle transformation
{
    // Validate that we can evenly divide the embedding dimension across heads
    if (d_model_ % n_heads_ != 0) {
        throw std::invalid_argument("d_model must be divisible by number of heads");
    }

    // Initialize all attention heads - each gets the same input but will learn different representations
    Initialize_All_Heads();

    // Initialize the output projection matrix that combines all head outputs
    Initialize_Output_Projection_Weights();
}

// Main interface - runs the complete multi-head attention computation
void Multi_Head_Attention:: Forward_Pass() {
    Compute_All_Head_Outputs();         // Run each attention head independently
    Concatenate_Attention_Heads();     // Combine all head outputs into one tensor
    Apply_Output_Projection();        // Final linear transformation to d_model dimensions
}

// Getter for the final multi-head attention output.This is a copy. We prefer copies for now.
Tensor Multi_Head_Attention::Get_Output() const {
    return final_output_;
}



// Step 1: Initialize all attention heads with appropriate input slices
void Multi_Head_Attention::Initialize_All_Heads() {
    attention_heads_.clear();
    attention_heads_.reserve(n_heads_);

    // Pass the full input to the attention heads. They will divide the dimensions as needed.
    for (int head_idx = 0; head_idx < n_heads_; ++head_idx) {
        // This allows different heads to attend to different types of relationships
        attention_heads_.emplace_back(batched_input_, d_k_);
    }
}

// Step 2: Initialize the output projection matrix using Xavier initialization
void Multi_Head_Attention::Initialize_Output_Projection_Weights() {
    // Create temporary matrix for Xavier initialization (same weights across all batches)
    Matrix temp_W_o(d_model_, d_model_, 0.0f);
    Matrix_Xavier_Uniform(temp_W_o);

    // Copy the same weight matrix to all batch channels
    // This ensures consistent behavior across batches during training
    for (int batch_idx = 0; batch_idx < batch_size_; ++batch_idx) {
        W_o_.Set_Channel_Matrix(temp_W_o, batch_idx);
    }
}

// Step 3: Run forward pass through each attention head independently
void Multi_Head_Attention::Compute_All_Head_Outputs() {
    // Each attention head computes its own query/key/value projections and attention weights
    // This parallelism allows the model to attend to different aspects simultaneously
    for (int head_idx = 0; head_idx < n_heads_; ++head_idx) {
        attention_heads_[head_idx].Forward_Pass();
    }
}

// Step 4: Concatenate outputs from all attention heads
// Concatenate multiple tensors along the column dimension (for attention heads)
void Multi_Head_Attention::Concatenate_Attention_Heads() {
    if (attention_heads_.empty()) {
        throw std::invalid_argument("Cannot concatenate empty attention heads vector");
    }

    // Collect all outputs once
    std::vector<Tensor> head_outputs;
    head_outputs.reserve(attention_heads_.size());
    for (const auto& head : attention_heads_) {
        head_outputs.push_back(head.Get_Output());
    }

    // Validate dimensions and calculate total columns
    int rows = head_outputs[0].rows();
    int depth = head_outputs[0].depth();
    int total_columns = 0;

    for (const auto& output : head_outputs) {
        if (output.rows() != rows || output.depth() != depth) {
            throw std::invalid_argument("All attention heads must have same sequence length (rows) and batch size (depth)");
        }
        total_columns += output.columns();
    }

    // Ensure concatenated_output_ has the right dimensions
    if (concatenated_output_.rows() != rows ||
        concatenated_output_.columns() != total_columns ||
        concatenated_output_.depth() != depth) {
        concatenated_output_ = Tensor(rows, total_columns, depth);
    }
    else{
        throw std::invalid_argument("The concatenated_output_ dimensions are not suitable for the concatenation operation ");
    }

    for (int d = 0; d < depth; ++d) {           // For each batch
        for (int r = 0; r < rows; ++r) {        // For each sequence position
            int dest_col = 0;

            for (const auto& head_output : head_outputs) {     // For each attention head output
                for (int c = 0; c < head_output.columns(); ++c) {  // For each feature in the head
                    concatenated_output_(r, dest_col, d) = head_output(r, c, d);
                    dest_col++;
                }
            }
        }
    }
}

// Step 5: Apply final linear projection to transform concatenated features
void Multi_Head_Attention::Apply_Output_Projection() {
    // Transform concatenated head outputs back to d_model dimensions
    // This allows the model to learn how to best combine information from different heads
    // Mathematical operation: final_output = concatenated_output * W_o
    Tensor_Multiply_Tensor(final_output_, concatenated_output_, W_o_);
}

