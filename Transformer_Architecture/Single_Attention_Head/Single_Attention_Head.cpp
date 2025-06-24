#include "Single_Attention_Head.h"
//
// Created by rakib on 23/6/2025.
//

#include "Single_Attention_Head.h"


//Upstream input tensor pouring into the attention block has the shape [sequence length (rows), d_model (columns), batch_number(depth/channel)]



Single_Attention_Head::Single_Attention_Head(const Tensor &batched_input, int d_k)
    :batched_input_(batched_input),sequence_length_(batched_input.rows()),d_model_(batched_input.columns()),batch_size_(batched_input.depth()),
    d_k_(d_k), W_q_(d_model_,d_k_,batch_size_),W_k_(d_model_,d_k_,batch_size_),W_v_(d_model_,d_k_,batch_size_),
    Q_(sequence_length_,d_k_,batch_size_),K_(sequence_length_,d_k_,batch_size_),V_(sequence_length_,d_k_,batch_size_),
    attention_scores_(sequence_length_,sequence_length_,batch_size_),attention_weights_(sequence_length_,sequence_length_,batch_size_),
    output_(sequence_length_,d_k_,batch_size_)
    {}







// The purpose of this function is to initialize the weights.
void Single_Attention_Head:: Initialize_Weights(){
    //We did not use the Tensor Xavier because it would not be the same for all batches.
    //Basically, we copied the 'Xavier'ed' weights from one slice(channel/depth) of the Tensor to all of its depth

    //Temporary buffer to store the weights
    Matrix W_q(d_model_,d_k_,0.0f);
    Matrix W_k(d_model_,d_k_,0.0f);
    Matrix W_v(d_model_,d_k_,0.0f);

    Matrix_Xavier_Uniform(W_q);
    Matrix_Xavier_Uniform(W_k);
    Matrix_Xavier_Uniform(W_v);

    for (int i=0;i<batch_size_;i++){
        W_q_.Set_Channel_Matrix(W_q,i);
        W_k_.Set_Channel_Matrix(W_k,i);
        W_v_.Set_Channel_Matrix(W_v,i);
    }
}


//Step 1: Projection. 'Mix' the input with the weights.
void Single_Attention_Head:: Compute_Projections() {
    // Need matrix multiplication: Input × Weights
    // Input: [sequence_length, d_model, batch_size]
    // Weights: [d_model, d_k, batch_size]
    // Result: [sequence_length, d_k, batch_size]. This dimension is simply a result of the standard matrix multiplication.

    //No need to initialize memory for Q_, K_ and V_ since they were already initialized in the constructor.

    // Transform input into three different representations
    Tensor_Multiply_Tensor(Q_, batched_input_, W_q_);  // Queries: "What am I looking for?      --> The question"
    Tensor_Multiply_Tensor(K_, batched_input_, W_k_);  // Keys: "What can I provide?    --> The 'lookup tables'"
    Tensor_Multiply_Tensor(V_, batched_input_, W_v_);  // Values: "What actual information do I contain?    --> The actual content of the 'lookup tables'"

    //Our excellent tensor library makes this simple with less code.
}

//Step 2: Compute Attention Scores. Basically a relevance map.
void Single_Attention_Head::Compute_Attention_Scores() {
    //This measures the relevance between queries and keys. A high value mathematically between two vectors means that they are similar.
    //Dot products are used to compute the relevance scores. The 'attention scores' are stored in the 'attention_scores_' tensor.
    //This is why dot products are useful.It gives us a mathematical way to represent 'similarity' between two vectors.

    //Allocate memory for transposed K tensor. This automatically fetches and initializes the tensor dimensions.
    //This special function automatically initializes a tensor by determining the tensor dimensions of any supported operation in our library.
    Tensor K_transposed = std::get<Tensor>(Memory_Allocation(Single_Tensor_Dependent_Operations::Tensor_Transpose, K_));

    // Actual transpose operation
    Tensor_Transpose(K_transposed, K_);

    // Compute Q × K^T - This measures similarity between queries and keys. Attention scores memory allocation is not needed since it was already allocated
    //First we make a temporary tensor. Look here that we did not have to manually calculate the memory allocation/dimensions for this tensor.
    Tensor initial_attention_scores= std::get<Tensor>(Memory_Allocation(Multi_Tensor_Dependent_Operations::Tensor_Multiply_Tensor, Q_, K_transposed));

    Tensor_Multiply_Tensor(initial_attention_scores, Q_, K_transposed);

    // Scale by sqrt(d_k) for numerical stability
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(d_k_));

    //The temporary allocation was needed so that the destination tensor and the source tensor are not the same (will create bugs).
    //We need to multiply the attention scores by the scale factor.
    Tensor_Multiply_Scalar_ElementWise(attention_scores_, initial_attention_scores, scale_factor);
}

//Step 3: Apply Softmax. This is a function that makes the d_k dimension of the attention scores into a probability distribution which sums to one.
void Single_Attention_Head:: Apply_Softmax_To_Attention_Scores() {
    // Convert attention scores to attention weights using softmax
    // attention_scores_: [sequence_length, sequence_length, batch_size]
    // Result: attention_weights_: [sequence_length, sequence_length, batch_size]

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

            // Normalize to probabilities
            for (int key_pos = 0; key_pos < sequence_length_; ++key_pos) {
                attention_weights_(query_pos, key_pos, batch_number) /= exp_sum;
            }
        }
    }
}

//Step 4: Compute Attention Output
void Single_Attention_Head:: Compute_Attention_Output() {
    // Multiply attention weights by values to get context-aware representations
    // attention_weights_: [sequence_length, sequence_length, batch_size]
    // V_: [sequence_length, d_k, batch_size]
    // Result: output_: [sequence_length, d_k, batch_size]
    Tensor_Multiply_Tensor(output_, attention_weights_, V_);
}
void Single_Attention_Head:: Forward_Pass() {
        Initialize_Weights();
        Compute_Projections();
        Compute_Attention_Scores();
        Apply_Softmax_To_Attention_Scores();
        Compute_Attention_Output();
    }

Tensor Single_Attention_Head::Get_Output() const {
    return output_;
}




//a higher d_k value indicates greater attention head coverage.






