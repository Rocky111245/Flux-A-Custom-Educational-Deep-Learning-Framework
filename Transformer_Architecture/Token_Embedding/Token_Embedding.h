//
// Created by rakib on 15/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKEN_EMBEDDING_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKEN_EMBEDDING_H

#include <vector>
#include "MatrixLibrary.h"
#include "Tensor_Library/Tensor_Library.h"
#include <stdexcept>
#include <vector>

//---------------------------------------------------------------------[Purpose]---------------------------------------------------------------------//
// This class implements the token embedding layer of a transformer model. It maps discrete token IDs to continuous embedding vectors.
//--------------------------------------------------------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------[Responsibility]------------------------------------------------------------------//
// - This class is responsible only for creating, storing, retrieving the token embedding matrix. It also updates the embedding matrix.
// - It does NOT handle vocabulary generation or tokenization.That is handled externally by the Tokenizer.
//--------------------------------------------------------------------------------------------------------------------------------------------------//

//--------------------------------------------------------------------[Matrix Details]----------------------------------------------------------------//
// - The embedding matrix is of shape [d_vocab, d_model]:
//      • d_vocab  = number of unique tokens (vocabulary size)
//      • d_model  = dimensionality of each token's embedding vector
// - Each row in the matrix corresponds to a token ID: embedding[token_id] = vector<float, d_model>
// - The embedding matrix is initialized using Xavier uniform distribution to ensure stable learning.
//--------------------------------------------------------------------------------------------------------------------------------------------------//


class Token_Embedding {

public:
    //=================================================================================================================================//
    //                                                     CONSTRUCTOR                                                                 //
    //=================================================================================================================================//


    Token_Embedding( int d_vocab,int d_model);

    //=================================================================================================================================//
    //                                                   GETTER METHODS                                                               //
    //=================================================================================================================================//

    std::vector<float> Token_Embedding::Get_Token_Embedding_Vector(int token_id) const;
    float Get_Embedding_Value(int token_id, int dimension_position) const;
    Tensor Get_Sequence_Embedding_Tensor(const Tensor& final_token_tensors,int sequence_number) const;
    Tensor Get_Batch_Embedding_Tensor(const Tensor& final_token_tensors, int start_row_number,
    int end_row_number) const;
    int Get_Vocab_Size() const;
    int Get_Model_Dimension() const;

    //=================================================================================================================================//
    //                                                  SETTER METHODS                                                                //
    //=================================================================================================================================//


    void Set_Token_Embedding_Value(int token_id, int dimension_position, float updated_value);
    void Set_Token_Embedding_Vector(int token_id, const std::vector<float> &embedding_vector);

private:
    int d_model_;                    // Dimensionality of each embedding vector
    int d_vocab_;                    // Vocabulary size (number of unique tokens)
    Tensor embedding_matrix_;        // The embedding matrix [d_vocab x d_model]
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKEN_EMBEDDING_H