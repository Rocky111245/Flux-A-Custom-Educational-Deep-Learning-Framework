//
// Created by rakib on 15/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_FIXED_POSITIONAL_ENCODING_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_FIXED_POSITIONAL_ENCODING_H
class Tensor;

Tensor Compute_Fixed_Sinusoidal_Encodings_Single_Sequence(int sequence_length, int d_model);
Tensor Compute_Fixed_Sinusoidal_Encodings_Single_Batch(int sequence_length,int d_model, int number_of_sequences);
#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_FIXED_POSITIONAL_ENCODING_H
