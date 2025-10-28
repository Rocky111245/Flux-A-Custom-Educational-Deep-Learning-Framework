//
// Created by rakib on 25/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_RESIDUAL_STREAM_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_RESIDUAL_STREAM_H

#include "tensor-library/TensorLibrary.h"

// Residual stream layout: [sequence_length, d_model, batch_size]
class ResidualStream {
public:
    explicit ResidualStream(Tensor upstream_embedded_batched_tensor);

    Tensor Clone() const;

    // Elementwise add: input must match current shape
    void Add_To_Residual(const Tensor& input);

    // Replace the stream; shape must match current shape
    void Reset(const Tensor& new_stream);

    // Accessors
    const Tensor& Get_Stream() const noexcept { return residual_stream_; }
    int Get_Batch_Size()      const noexcept { return residual_stream_.depth(); }
    int Get_d_model()         const noexcept { return residual_stream_.columns(); }
    int Get_Sequence_Length() const noexcept { return residual_stream_.rows(); }

private:
    Tensor residual_stream_; // [seq_len, d_model, batch_size]
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_RESIDUAL_STREAM_H