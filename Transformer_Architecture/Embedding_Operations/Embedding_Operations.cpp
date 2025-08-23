//
// Created by rakib on 16/6/2025.
//

#include "Embedding_Operations.h"
#include "Tensor_Library/Tensor_Library.h"


//this class handles addition of the token_embedding and positional encodings

Tensor Embeddings_Add_Per_Sequence(const Tensor &token_embedding, const Tensor &fixed_positional_encoding){
    if (token_embedding.rows() != fixed_positional_encoding.rows() || token_embedding.columns() != fixed_positional_encoding.columns()){
        throw std::invalid_argument("Token embedding and fixed positional encoding must have the same dimensions.");
    }
    //create temp tensor
    Tensor destination_tensor(token_embedding.rows(),token_embedding.columns(),1);

    //this mixes up the token and positional encodings
    Tensor_Add_Tensor_ElementWise(destination_tensor, token_embedding, fixed_positional_encoding);
    return destination_tensor;
}
Tensor Embeddings_Add_Per_Batch(const Tensor &token_embedding, const Tensor &fixed_positional_encoding){

    if (token_embedding.rows() != fixed_positional_encoding.rows() || token_embedding.columns() != fixed_positional_encoding.columns()){
        throw std::invalid_argument("Token embedding and fixed positional encoding must have the same dimensions.");
    }
    //create temp tensor
    Tensor destination_tensor(token_embedding.rows(),token_embedding.columns(),token_embedding.depth());

    //this mixes up the token and positional encodings
    Tensor_Add_Tensor_ElementWise(destination_tensor, token_embedding, fixed_positional_encoding);
    return destination_tensor;
}
