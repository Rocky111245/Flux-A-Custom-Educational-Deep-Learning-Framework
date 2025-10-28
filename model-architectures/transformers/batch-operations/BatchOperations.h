//
// Created by Rocky170 on 8/14/2025.
//

#ifndef BATCH_OPERATIONS_H
#define BATCH_OPERATIONS_H
#include <vector>
#include "tensor-library/TensorLibrary.h"
#include "model-architectures/transformers/token_embedding/TokenEmbedding.h"


class Tensor;
struct batch_plan {
    const int desired_tokens_per_batch;
    const int minimum_batch_size;
    int sequences_per_batch;                  // sequences per batch
    int total_number_of_batches;                 // total number of batches
    int total_sequences;                       //total sequences present
    std::vector<int> start_index_of_batch;      // starting sequence index for each batch
    std::vector<int> seqs_in_batch;  // sequences in each batch (last may be smaller)
};

struct global_objects {
    TokenEmbedding token_embeddings;
};

struct hyper_parameters {
    int d_model;
};

void Batch_Tokenize(const Tensor& final_token_tensors,batch_plan &plan);
Tensor Batch_Prepare_Sequence_Embeddings (global_objects objects,const Tensor& final_token_tensors,const batch_plan &batch_plan,
    const hyper_parameters &hyper_parameters,int batch_number);



#endif //BATCH_OPERATIONS_H
