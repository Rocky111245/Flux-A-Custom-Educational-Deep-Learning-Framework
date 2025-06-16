//
// Created by rakib on 15/6/2025.
//

#include "Token_Embedding.h"
#include "NLP Utilities/Tokenizer.h"
#include <stdexcept>

//=================================================================================================================================//
//                                                     CONSTRUCTOR                                                                 //
//=================================================================================================================================//

Token_Embedding::Token_Embedding(int d_model, int d_vocab)
        : d_model_(d_model), d_vocab_(d_vocab), embedding_matrix_(d_vocab, d_model) {

    // Initialize the embedding matrix using Xavier uniform distribution for stable learning
    Matrix_Xavier_Uniform(embedding_matrix_);
}

//=================================================================================================================================//
//                                                   GETTER METHODS                                                               //
//=================================================================================================================================//

const float Token_Embedding::Get_Embedding_Value(int token_id, int dimension_position) const {
    return embedding_matrix_(token_id, dimension_position);
}

const std::vector<float> Token_Embedding::Get_Embedding_Vector(int token_id) const {
    std::vector<float> embedding_vector;
    embedding_vector.reserve(embedding_matrix_.columns());

    // token_id is directly related to row number, we always start from 0
    for (int i = 0; i < embedding_matrix_.columns(); i++) {
        embedding_vector.emplace_back(embedding_matrix_(token_id, i));
    }
    return embedding_vector;
}

const int Token_Embedding::Get_Vocab_Size() const {
    return d_vocab_;
}

const int Token_Embedding::Get_Model_Dimension() const {
    return d_model_;
}

//=================================================================================================================================//
//                                                  BATCH PROCESSING                                                              //
//=================================================================================================================================//

Matrix Token_Embedding::Get_Embedding_Matrix_For_Single_Batch(const std::vector<int> &sequence) const {
    Matrix matrix(sequence.size(), d_model_);

    for (int i = 0; i < sequence.size(); i++) {
        for (int j = 0; j < embedding_matrix_.columns(); j++) {
            matrix(i, j) = embedding_matrix_(sequence[i], j);  // embedding_matrix_ is [vocab_size, d_model]
        }
    }
    return matrix;
}

Tensor Token_Embedding::Get_Embedding_Matrix_For_All_Batches(const Matrix &batch_matrix) const {
    // Create tensor with custom layout: T(sequence_length, d_model, batch_index)
    // Maps to Tensor(rows, columns, depth) structure
    Tensor tensor_all_batches(batch_matrix.columns(), d_model_, batch_matrix.rows());

    for (int batch_index = 0; batch_index < batch_matrix.rows(); batch_index++) {
        for (int sequence_number = 0; sequence_number < batch_matrix.columns(); sequence_number++) {
            int token_id = batch_matrix(batch_index, sequence_number);

            for (int embedding_dimension_index = 0; embedding_dimension_index < d_model_; embedding_dimension_index++) {
                tensor_all_batches(sequence_number, embedding_dimension_index, batch_index) =
                        embedding_matrix_(token_id, embedding_dimension_index);
            }
        }
    }
    return tensor_all_batches;

    // NOTE: If you have difficulty understanding this function, remember that our custom Tensor data structure
    // has the structure Tensor(rows, columns, depth) where column is the fastest mover,
    // second is the row, then the depth lastly. This is intuitive and resembles our Matrix class->Matrix(rows, columns)
}

//=================================================================================================================================//
//                                                  SETTER METHODS                                                                //
//=================================================================================================================================//

void Token_Embedding::Set_Embedding_Value(int token_id, int dimension_position, float updated_value) {
    embedding_matrix_(token_id, dimension_position) = updated_value;
}

void Token_Embedding::Set_Embedding_Vector(int token_id, const std::vector<float> &embedding_vector) {
    if (embedding_vector.size() != embedding_matrix_.columns()) {
        throw std::invalid_argument("Embedding vector size mismatch with d_model.");
    }

    for (int i = 0; i < embedding_vector.size(); i++) {
        embedding_matrix_(token_id, i) = embedding_vector[i];
    }
}