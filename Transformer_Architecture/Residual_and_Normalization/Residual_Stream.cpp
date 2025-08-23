//
// Created by rakib on 25/6/2025.


#include "Residual_Stream.h"

#include <utility>
#include "Tensor_Library/Tensor_Library.h"

// The residual stream is the single, global communication bus that links every layer of a transformer together.
// It is a 'stream' that every layer can read from and write to by adding small “deltas” or changes.
// Originates after the token+optionally positional embeddings
// The usage will occur right after the Batch_Operations.cpp returning a tenor that looks like [n_tokens_in_a_sequence,d_model,n_sequences]

//The residual stream is the unbroken backbone of the transformer. It starts life as the token embedding,
//travels unchanged in its core path, and collects contributions from every attention head and every FFN neuron across
//all layers, enabling a linear superposition of features that the final head decodes into predictions.


//Updated on 23/08/2025
Residual_Stream::Residual_Stream(Tensor upstream_embedded_batched_tensor)
    : residual_stream_(std::move(upstream_embedded_batched_tensor))
{
    if (residual_stream_.rows()    <= 0 ||
        residual_stream_.columns() <= 0 ||
        residual_stream_.depth()   <= 0) {
        throw std::invalid_argument(
            "Residual_Stream: upstream tensor must have positive dimensions, got ");
        }
}

// Clone a deep copy for processing (e.g., LayerNorm → Attention path)
Tensor Residual_Stream::Clone() const {
    assert(residual_stream_.rows()    > 0 &&
           residual_stream_.columns() > 0 &&
           residual_stream_.depth()   > 0 && "Residual_Stream::Clone()  empty tensor");

    // Release safety (if misuse happens)
    if (residual_stream_.rows()    <= 0 ||
        residual_stream_.columns() <= 0 ||
        residual_stream_.depth()   <= 0) {
        throw std::logic_error("Residual_Stream::Clone() invalid state: Invalid Dimensions for residual stream");
        }

    return Tensor{residual_stream_}; // copy
}

// Add a new tensor to the residual stream (output of attention/FFN). It is always element-wise addition
void Residual_Stream::Add_To_Residual(const Tensor& input) {
    if (input.rows()    != residual_stream_.rows() ||
        input.columns() != residual_stream_.columns() ||
        input.depth()   != residual_stream_.depth()) {
        throw std::invalid_argument(
            "Add_To_Residual: shape mismatch, residual= + "
            " input=" );
        }
    residual_stream_ += input; // element-wise in-place add
}

// Reset the residual stream (e.g., for new input).
void Residual_Stream::Reset(const Tensor& new_stream) {
    if (new_stream.rows()    != residual_stream_.rows() ||
        new_stream.columns() != residual_stream_.columns() ||
        new_stream.depth()   != residual_stream_.depth()) {
        throw std::invalid_argument(
            "Reset: shape mismatch, expected ");
        }
    residual_stream_ = new_stream;
}


