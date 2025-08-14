//
// Created by rakib on 16/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_EMBEDDING_OPERATIONS_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_EMBEDDING_OPERATIONS_H

class Tensor;

Tensor Embeddings_Add_Per_Sequence(const Tensor &token_embedding, const Tensor &fixed_positional_encoding);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_EMBEDDING_OPERATIONS_H
