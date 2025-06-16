//
// Created by rakib on 15/6/2025.
//

#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKEN_EMBEDDING_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKEN_EMBEDDING_H

#include <vector>
#include "MatrixLibrary.h"
#include "Tensor_Library/Tensor_Library.h"

//---------------------------------------------------------------------[Purpose]---------------------------------------------------------------------//
// This class implements the token embedding layer of a transformer model. It maps discrete token IDs to continuous embedding vectors.
//--------------------------------------------------------------------------------------------------------------------------------------------------//

//------------------------------------------------------------------[Responsibility]------------------------------------------------------------------//
// - This class is responsible only for creating, storing, retrieving and modifying the token embedding matrix.
// - It does NOT handle vocabulary generation or tokenization. That is handled externally by the Tokenizer class.
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


    Token_Embedding(int d_model, int d_vocab);

    //=================================================================================================================================//
    //                                                   GETTER METHODS                                                               //
    //=================================================================================================================================//

    const float Get_Embedding_Value(int token_id, int dimension_position) const;


    const std::vector<float> Get_Embedding_Vector(int token_id) const;


    const int Get_Vocab_Size() const;


    const int Get_Model_Dimension() const;

    //=================================================================================================================================//
    //                                                  BATCH PROCESSING                                                              //
    //=================================================================================================================================//


    Matrix Get_Embedding_Matrix_For_Single_Batch(const std::vector<int> &sequence) const;


    Tensor Get_Embedding_Matrix_For_All_Batches(const Matrix &batch_matrix) const;

    //=================================================================================================================================//
    //                                                  SETTER METHODS                                                                //
    //=================================================================================================================================//


    void Set_Embedding_Value(int token_id, int dimension_position, float updated_value);


    void Set_Embedding_Vector(int token_id, const std::vector<float> &embedding_vector);

private:
    //=================================================================================================================================//
    //                                                  MEMBER VARIABLES                                                              //
    //=================================================================================================================================//

    int d_model_;                    // Dimensionality of each embedding vector
    int d_vocab_;                    // Vocabulary size (number of unique tokens)
    Matrix embedding_matrix_;        // The embedding matrix [d_vocab x d_model]
};

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKEN_EMBEDDING_H