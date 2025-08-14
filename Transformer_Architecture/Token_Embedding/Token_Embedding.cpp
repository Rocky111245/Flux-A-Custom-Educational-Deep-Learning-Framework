
//  Created by rakib on 15/6/2025.
// Updated on 08/14/2025
//  ──────────────────────────────────────────────────────────────────────────────
//  Token_Embedding.cpp
//  ──────────────────────────────────────────────────────────────────────────────

#include "Token_Embedding.h"
#include "Transformer_Architecture/Tokenizer/Tokenizer.h"


//  =============================================================================
//  CONSTRUCTOR
//  =============================================================================
Token_Embedding::Token_Embedding(const int d_vocab, const int d_model)
    : d_model_(d_model),
      d_vocab_(d_vocab),
      embedding_matrix_(d_vocab, d_model, 1)
{
    // Xavier uniform initialization for stable learning
    embedding_matrix_.Tensor_Xavier_Uniform_Share_Across_Depth();
}

//  =============================================================================
//  GETTER METHODS
//  =============================================================================

//gets the embedding vector of a SINGLE token
std::vector<float> Token_Embedding::Get_Token_Embedding_Vector(const int token_id) const
{
    assert(token_id < d_vocab_ && "Token ID is not present in the vocabulary, out of bounds");

    std::vector<float> result;
    result.reserve(d_model_);
    for (int i = 0; i < d_model_; ++i)
        result.emplace_back(embedding_matrix_(token_id, i, 0));

    return result;
}


// Gets the full embedding tensor for a single sequence in the format [sequence_length[i], d_model, 1]
// Tensor is 2D here
Tensor Token_Embedding::Get_Sequence_Embedding_Tensor(const Tensor& token_tensor, const int sequence_number) const {
    // Ensure the sequence number is valid
    if (sequence_number >= token_tensor.rows() ) {
        throw std::invalid_argument("Sequence number out of bounds in Token_Embedding::Get_Sequence_Embedding_Tensor");
    }

    int token_id=0;
    // Create a tensor that holds the full sequence in its rows
    Tensor out(token_tensor.columns(), d_model_, 1);

    for (int i = 0; i < token_tensor.columns(); ++i) {
        token_id = token_tensor(sequence_number, i, 0);

        assert(token_id < d_vocab_ && "Token ID out of bounds");

        for (int j = 0; j < d_model_; ++j) {
            out(i, j, 0) = embedding_matrix_(token_id, j, 0);
        }
    }
    return out;
}




int Token_Embedding::Get_Vocab_Size()  const { return d_vocab_; }
int Token_Embedding::Get_Model_Dimension() const { return d_model_; }

//  =============================================================================
//  SETTER METHODS
//  =============================================================================
void Token_Embedding::Set_Token_Embedding_Value(const int token_id,const int dimension_position,const float updated_value)
{
    if (token_id >= d_vocab_ || dimension_position >= d_model_)
        throw std::invalid_argument("token_id or dimension_position out of range in Token_Embedding::Set_Token_Embedding_Value");

    embedding_matrix_(token_id, dimension_position, 0) = updated_value;
}

void Token_Embedding::Set_Token_Embedding_Vector(const int token_id,const std::vector<float>& embedding_vector)
{
    if (embedding_vector.size() != static_cast<std::size_t>(d_model_)) {
        throw std::invalid_argument("Embedding vector size mismatch with d_model");
    }

    assert(embedding_vector.size() == static_cast<std::size_t>(embedding_matrix_.columns()));

    for (int i = 0; i < embedding_vector.size(); ++i) {
        embedding_matrix_(token_id, i, 0) = embedding_vector[i];
    }
}