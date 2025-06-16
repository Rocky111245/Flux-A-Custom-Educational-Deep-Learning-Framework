//
// Created by rakib on 16/6/2025.
//

#include "Embedding_Operations.h"
#include "MatrixLibrary.h"
#include "Tensor_Library/Tensor_Library.h"
#include "Transformer_Architecture/Token_Embedding/Token_Embedding.h"
#include "Transformer_Architecture/Positional_Embeddings/Fixed_Positional_Encoding.h"



void Add_Embeddings_Per_Batch(Matrix &destination_matrix, const Matrix &token_embedding, const Matrix &fixed_positional_encoding){
    if (token_embedding.rows() && token_embedding.columns()!= fixed_positional_encoding.rows() && fixed_positional_encoding.columns()){
        throw std::invalid_argument("Token embedding and fixed positional encoding must have the same number of columns.");
    }
    Matrix_Add(destination_matrix,token_embedding,fixed_positional_encoding);
}


void Add_Embeddings_All_Batches(Tensor &destination_matrix, const Tensor &token_embedding, const Matrix &fixed_positional_encoding){

}
