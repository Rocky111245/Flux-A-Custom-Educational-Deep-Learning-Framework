#include "Single_Attention_Head.h"
//
// Created by rakib on 23/6/2025.
//

#include "Single_Attention_Head.h"


//Upstream input tensor pouring into the attention block has the shape [sequence length (rows), d_model (columns), batch_number(depth/channel)]



Single_Attention_Head::Single_Attention_Head(const Tensor &residual_stream_input, const int d_k, const bool masked_attention)
    :residual_stream_input_copy_(residual_stream_input),sequence_length_(residual_stream_input.rows()),d_model_(residual_stream_input.columns()),batch_size_(residual_stream_input.depth()),
    d_k_(d_k), W_q_(d_model_,d_k_,batch_size_),W_k_(d_model_,d_k_,batch_size_),W_v_(d_model_,d_k_,batch_size_),
    Q_(sequence_length_,d_k_,batch_size_),K_(sequence_length_,d_k_,batch_size_),V_(sequence_length_,d_k_,batch_size_),
    attention_scores_(sequence_length_,sequence_length_,batch_size_),attention_weights_(sequence_length_,sequence_length_,batch_size_),
    output_(sequence_length_,d_k_,batch_size_), masked_attention_(masked_attention)
{
    if (d_k <= 0 || d_model_ <= 0 || sequence_length_ <= 0 || batch_size_ <= 0) {
        throw std::invalid_argument("Invalid tensor dimensions or d_k in Single_Attention_Head constructor.");
    }
    Initialize_Weights();
}







// The purpose of this function is to initialize the weights.
void Single_Attention_Head:: Initialize_Weights(){
    //This Tensor Xavier only Xaviers for a single depth. So we just broadcast the matrix into the tensors since they have to be the same.
    W_k_.Tensor_Xavier_Uniform_Share_Across_Depth();
    W_q_.Tensor_Xavier_Uniform_Share_Across_Depth();
    W_v_.Tensor_Xavier_Uniform_Share_Across_Depth();
}


//Step 1: Projection. 'Mix' the input with the weights. We use a low rank matrix to compress d_model to d_k
void Single_Attention_Head:: Compute_Projections() {
    // Need matrix multiplication: Input × Weights
    // Input: [sequence_length, d_model, batch_size]
    // Weights: [d_model, d_k, batch_size]
    // Result: [sequence_length, d_k, batch_size]. This dimension is simply a result of the standard matrix multiplication.

    //No need to initialize memory for Q_, K_ and V_ since they were already initialized in the constructor.
    assert(residual_stream_input_copy_.columns() == W_q_.rows() &&
            "Single_Attention_Head::Compute_Projections -> Mismatch: input.columns != W_q.rows");

    assert(residual_stream_input_copy_.rows() == Q_.rows() &&
           "Single_Attention_Head::Compute_Projections -> Mismatch: input.rows != Q_.rows");

    assert(W_q_.columns() == Q_.columns() &&
           "Single_Attention_Head::Compute_Projections -> Mismatch: W_q.columns != Q_.columns");


    // Transform input into three different representations
    Tensor_Multiply_Tensor(Q_, residual_stream_input_copy_, W_q_);  // Queries: "What am I looking for?      --> The question"
    Tensor_Multiply_Tensor(K_, residual_stream_input_copy_, W_k_);  // Keys: "What can I provide?    --> The 'lookup tables'"
    Tensor_Multiply_Tensor(V_, residual_stream_input_copy_, W_v_);  // Values: "What actual information do I contain?    --> The actual content of the 'lookup tables'"

    //Our excellent tensor library makes this simple with less code.
}

//Step 2: Compute Attention Scores. Basically a relevance map.
void Single_Attention_Head::Compute_Attention_Scores() {
    //This measures the relevance between queries and keys. A high value mathematically between two vectors means that they are similar.
    //Dot products are used to compute the relevance scores. The 'attention scores' are stored in the 'attention_scores_' tensor.
    //This is why dot products are useful.It gives us a mathematical way to represent 'similarity' between two vectors.
    assert(Q_.columns() == K_.columns() &&
               "Single_Attention_Head::Compute_Attention_Scores -> Q_.columns != K_.columns");

    assert(Q_.depth() == K_.depth() &&
           "Single_Attention_Head::Compute_Attention_Scores -> Q_.depth != K_.depth");

    assert(attention_scores_.rows() == Q_.rows() &&
           "Single_Attention_Head::Compute_Attention_Scores -> attention_scores_.rows != Q_.rows");

    assert(attention_scores_.columns() == K_.rows() &&
           "Single_Attention_Head::Compute_Attention_Scores -> attention_scores_.columns != K_.rows");

    assert(attention_scores_.depth() == Q_.depth() &&
           "Single_Attention_Head::Compute_Attention_Scores -> attention_scores_.depth != Q_.depth");


    //Allocate memory for transposed K tensor. This automatically fetches and initializes the tensor dimensions.
    Tensor K_transposed(K_.columns(),K_.rows(),K_.depth());

    // Actual transpose operation
    Tensor_Transpose(K_transposed, K_);

    // Compute Q × K^T - This measures similarity between queries and keys. Attention scores memory allocation is not needed since it was already allocated
    //First we make a temporary tensor.
    Tensor initial_attention_scores(Q_.rows(),K_transposed.columns(),Q_.depth());

    Tensor_Multiply_Tensor(initial_attention_scores, Q_, K_transposed);

    // Scale by sqrt(d_k) for numerical stability
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k_));

    //The temporary allocation was needed so that the destination tensor and the source tensor are not the same (will create bugs).
    //We need to multiply the attention scores by the scale factor.
    Tensor_Multiply_Scalar_ElementWise(attention_scores_, initial_attention_scores, scale_factor);
}

//If we want Masked Attention,apply this
void Single_Attention_Head:: Apply_Causal_Mask() {
    const int rows=attention_scores_.rows();
    const int columns=attention_scores_.columns();
    const int depth=attention_scores_.depth();
    const float mask = -std::numeric_limits<float>::infinity();

    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < rows; ++j) {
            for (int k = j + 1; k < columns; ++k) {
                attention_scores_(j, k, i) = mask;
            }
        }
    }

}

//Step 3: Apply Softmax. This is a function that makes the d_k dimension of the attention scores into a probability distribution which sums to one.
void Single_Attention_Head:: Apply_Softmax_To_Attention_Scores() {
    // Convert attention scores to attention weights using softmax
    // attention_scores_: [sequence_length, sequence_length, batch_size]
    // Result: attention_weights_: [sequence_length, sequence_length, batch_size]
    assert(attention_scores_.depth() == attention_weights_.depth() &&
              "Single_Attention_Head::Apply_Softmax -> Depth mismatch between scores and weights");

    assert(attention_scores_.rows() == attention_weights_.rows() &&
           "Single_Attention_Head::Apply_Softmax -> Row mismatch between scores and weights");

    assert(attention_scores_.columns() == attention_weights_.columns() &&
           "Single_Attention_Head::Apply_Softmax -> Column mismatch between scores and weights");
    if (masked_attention_) {
        Apply_Causal_Mask();
    }
    for (int batch_number = 0; batch_number < batch_size_; ++batch_number) {
        for (int query_pos = 0; query_pos < sequence_length_; ++query_pos) {
            // Find max for numerical stability across this query's row
            float max_score = -std::numeric_limits<float>::infinity();
            for (int key_pos = 0; key_pos < sequence_length_; ++key_pos) {
                max_score = std::max(max_score, attention_scores_(query_pos, key_pos, batch_number));
            }

            // Compute exponential and sum
            float exp_sum = 0.0f;
            for (int key_pos = 0; key_pos < sequence_length_; ++key_pos) {
                float exp_val = std::exp(attention_scores_(query_pos, key_pos, batch_number) - max_score);
                attention_weights_(query_pos, key_pos, batch_number) = exp_val;
                exp_sum += exp_val;
            }
            assert(exp_sum > 0.0f && "Single_Attention_Head::Apply_Softmax -> Exponential sum is zero or negative");

            // Normalize to probabilities
            for (int key_pos = 0; key_pos < sequence_length_; ++key_pos) {
                attention_weights_(query_pos, key_pos, batch_number) /= exp_sum;
            }
        }
    }
}

//Step 4: Compute Attention Output/ ATTENTION
void Single_Attention_Head:: Compute_Attention_Output() {
    // Multiply attention weights by values to get context-aware representations
    // attention_weights_: [sequence_length, sequence_length, batch_size]
    // V_: [sequence_length, d_k, batch_size]
    // Result: output_: [sequence_length, d_k, batch_size]

    assert(attention_weights_.columns() == V_.rows() &&
         "Single_Attention_Head::Compute_Attention_Output -> attention_weights_.columns != V_.rows");

    assert(attention_weights_.depth() == V_.depth() &&
           "Single_Attention_Head::Compute_Attention_Output -> Depth mismatch");
    Tensor_Multiply_Tensor(output_, attention_weights_, V_);
}
void Single_Attention_Head:: Forward_Pass() {

        Compute_Projections();
        Compute_Attention_Scores();
        Apply_Softmax_To_Attention_Scores();
        Compute_Attention_Output();
}

// This gives a copy
Tensor Single_Attention_Head::Get_Output_Clone() const {
    return output_;
}
//This gives read-only access
const Tensor& Single_Attention_Head::Get_Output_View() const {
    return output_;
}

//This allows mutability on return value
Tensor& Single_Attention_Head::Get_Output_Mutable()  {
    return output_;
}


//a higher d_k value indicates greater attention head coverage and lesser compression of dimensions from d_head.






