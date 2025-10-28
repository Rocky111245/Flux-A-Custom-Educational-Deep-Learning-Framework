//
// Created by Rocky170 on 8/14/2025.
//

#include "BatchOperations.h"

#include "EmbeddingOperations.h"
#include "FixedPositionalEncoding.h"


//Here we establish the strides for each batch and the number of batches.
//Basically we divide the sequences into batches using indices.
void Batch_Tokenize(const Tensor& final_token_tensors,batch_plan &plan)

{
    const int total_sequences = final_token_tensors.rows();
    const int seq_length      = final_token_tensors.columns();

    if (total_sequences <= 0 || seq_length <= 0) {
        throw std::invalid_argument("Invalid dataset shape, fix plan");
    }

    if (plan.minimum_batch_size <= 0) {
        throw std::invalid_argument("minimum_batch_size must be >= 1, fix plan");
    }


    // sequences per batch from token budget (truncate)
    int batch_size = plan.desired_tokens_per_batch / seq_length;

    // clamp & make usable
    if (batch_size <= 0) batch_size = 1;
    if (batch_size < plan.minimum_batch_size) batch_size = plan.minimum_batch_size;
    if (batch_size > total_sequences) batch_size = total_sequences;

    const int num_batches = (total_sequences + batch_size - 1) / batch_size; // ceil
    //suppose we got 20 sequences per batch (floored sequences_per_batch)
    // suppose total sequence is 1237
    // division results in ceiling of 1237/20= 61.85--> 62 (last answer)
    // total_number_of_batches gives us the maximum number of batches possible,although last batch would be smaller


    plan.sequences_per_batch  = batch_size;         //how many sequences makes a batch? Example:6 sequences(6 rows)
    plan.total_number_of_batches = num_batches;     //how many total batches are present?  Example: 62 total batches
    plan.start_index_of_batch.resize(num_batches);  // what is the start index of each batch? Example :0 for batch 1
    plan.seqs_in_batch.resize(num_batches);         // how many sequences are present in a batch? all same except last.
    plan.total_sequences = total_sequences;

    for (int b = 0; b < num_batches; ++b) {
        const int start = b * batch_size;
        const int rows  = std::min(batch_size, total_sequences - start); //second term for the very last batch
        plan.start_index_of_batch[b] = start;
        plan.seqs_in_batch[b] = rows;
    }


    int covered = 0;
    for (const int r : plan.seqs_in_batch) covered += r;
    assert(covered == total_sequences && "The number of sequences in the batch plan covered is not the same as the total"
                                         "sequences in the batch plan");

}
//this embeds a pure tokenized tensor to have token and positional embeddings together for a single batch
//making it ready as the input to the attention block.
//Basically suppose we choose batch 5, all the sequences (rows in Tensor&final_token_tensors will be embedded).
//Returns a new tensor

Tensor Batch_Prepare_Sequence_Embeddings (
    global_objects objects,
    const Tensor& final_token_tensors,
    const batch_plan &batch_plan,
    const hyper_parameters &hyper_parameters,
    const int batch_number               )


{

    if (batch_number>batch_plan.total_number_of_batches) {
        throw std::invalid_argument("Choose a batch below the total number of batches for embeddings");
    }

    const int start_index= batch_plan.start_index_of_batch[batch_number];
    assert(start_index==(batch_number*batch_plan.sequences_per_batch) && "This should match");
    int end_index=0;

    if (batch_number!=batch_plan.total_number_of_batches) { //we check that it isnt the last batch
        end_index= batch_plan.start_index_of_batch[batch_number+1];
    }
    else {
        end_index=batch_plan.total_sequences-batch_plan.start_index_of_batch[batch_number];
        assert(end_index==final_token_tensors.rows()-1 && "End_index DOES NOT MATCH THE NUMBER OF ROWS, 1 BY OFF ERROR \n"
                                                          && "End index= " && end_index );
    }
    assert(end_index>=start_index && "Wrong batch indexes, this should not be possible");

    const int number_of_sequences = end_index - start_index;
    //it is important that we return a tensor,this will be the birthplace of the residual stream and input
    //to the attention blocks
    // the embeddings are all in the form of [seq_size as rows,d_model as columns and batch size of 1]

    const Tensor token_embeddings_tensor = objects.token_embeddings.Get_Batch_Embedding_Tensor(final_token_tensors,
                                                                                               start_index, end_index);
    const Tensor positional_embeddings_tensor = Compute_Fixed_Sinusoidal_Encodings_Single_Batch(final_token_tensors.columns(),
        hyper_parameters.d_model, number_of_sequences);
    assert(token_embeddings_tensor.rows() == positional_embeddings_tensor.rows() && token_embeddings_tensor.columns() == positional_embeddings_tensor.columns()
    && token_embeddings_tensor.depth()==positional_embeddings_tensor.depth() &&
    "Token Embedding and Positional Embedding dimension mismatch in Tensor Batch_Prepare_Sequence_Embeddings");

    //this returns a full batch of tokens
    Tensor final_batched_token_tensor = Embeddings_Add_Per_Batch(token_embeddings_tensor, positional_embeddings_tensor);

    //embedded final batched token tensor have the dimensions [tokens in a sequence as the rows,d_model in columns,
    // sequences in the depth dimension].


    return final_batched_token_tensor ;
}






