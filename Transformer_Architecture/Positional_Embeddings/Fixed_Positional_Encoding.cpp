#include "Fixed_Positional_Encoding.h"
#include "Tensor_Library/Tensor_Library.h"
#include <cmath>

//---------------------------------------------------------------------------------------------------------//
// Description:
//     Generates fixed sinusoidal positional encodings as introduced in the "Attention is All You Need" paper.
//     This function fills a matrix with shape [max_sequence_length, d_model], where each row represents a
//     position in the input sequence and each column encodes a frequency component and in the case of a tensor
//     [max_sequence_length, d_model, 1] where a Tensor is Tensor(row,column,depth)
//
// Parameters:
//     - fixed_positional_encoding_tensor: Tensor to be populated with positional encodings
//     - max_sequence_length             : Maximum number of positions to encode.Usually the row of the tensor above.
//     - d_model                         : Dimensionality of each position vector (must match token embedding size)
//     - 1                               : For now supports one full sequence. Returns a tensor of positions for
//                                       : one sequence
// Notes:
//     - Even dimensions use sine; odd dimensions use cosine, with decreasing frequency as dimension increases.
//     - This matrix is fixed (non-learnable) and is meant to be added element-wise to token embeddings externally.
//       - It is ONLY computed once since there are not learnable parameters.
//
// Formula:
//     PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
//     PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
//
//---------------------------------------------------------------------------------------------------------//

// Core function that computes the sinusoidal positional encodings
Tensor Compute_Fixed_Sinusoidal_Encodings_Single_Sequence(const int sequence_length, const int d_model) {

//Resize just in case the tensor is wrongly initialized. Matrices/Tensors should be properly initialized;
//I do not want to have memory allocations inside the function.
//unless needed, but this is a necessity for now in the library. Will remove it later.

    Tensor fixed_positional_encoding_tensor(sequence_length,d_model,1);

    for (int pos = 0; pos < sequence_length; pos++) {
        for (int i = 0; i < d_model; i++) {
            // Calculate frequency using the correct formula: 2i/d_model for pairs
            int pair_index = i / 2;  // Which sine-cosine pair we're in
            float frequency_factor = 2.0f * (float)pair_index / (float)d_model;
            float frequency = 1.0f / std::pow(10000.0f, frequency_factor);

            if (i % 2 == 0) {
                // Even dimension: use sine
                fixed_positional_encoding_tensor(pos, i,0) = std::sin((float)pos * frequency);
            } else {
                // Odd dimension: use cosine
                fixed_positional_encoding_tensor(pos, i,0) = std::cos((float)pos * frequency);
            }
        }
    }
    return fixed_positional_encoding_tensor;
}



