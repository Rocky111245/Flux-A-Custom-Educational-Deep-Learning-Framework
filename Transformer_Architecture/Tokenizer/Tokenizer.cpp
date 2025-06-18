//
// Created by rakib on 18/6/2025.
//

#include "Tokenizer.h"
#include "MatrixLibrary.h"


//Implementation of Byte-Pair Encoding Algorithm


//This converts the corpus to a vector of strings having single characters every index
//This is because later they will be merged, and we will ultimately need a string vector
void Word_To_PreTokens(std::vector<std::string> &string_vector, const std::string& sentence) {
    int string_size = sentence.length();
    string_vector.emplace_back("_");

    for (int i = 0; i < string_size; i++) {
        if (std::isspace(sentence[i])) {

            string_vector.emplace_back("_");// New word start
        }
        else {
            string_vector.emplace_back(1, sentence[i]);
        }
    }
}


void Create_Vocabulary(const std::vector<std::string>&string_vector,std::unordered_set<std::string> &vocabulary){
    int pretoken_vector_size=string_vector.size();

    for(int i=0;i<pretoken_vector_size;i++){
        vocabulary.emplace(string_vector[i]);
    }
}



void Calculate_Bigram_Frequency(std::vector<std::string>& string_vector,
                                std::unordered_map<std::string,int>& bigram_counts) {

    for(int i=0;i<string_vector.size()-1;i++) {
        std::string bigram_string = string_vector[i] + string_vector[i + 1];
        bigram_counts[bigram_string]++;
    }
}


void Greedy_Bigram_Merging(std::vector<std::string>& string_vector, std::unordered_set<std::string> &vocabulary,
                           std::unordered_map<std::string,int>& bigram_counts, int max_vocabulary_size) {

    bool found_merge_condition = true;
    int max_value = 0;
    std::string key;
    std::string bigram_string;

    while (found_merge_condition && vocabulary.size()<max_vocabulary_size) {
        // Reset for this iteration
        found_merge_condition = false;
        max_value = 0;
        key.clear();

        // FIRST: Find the most frequent bigram
        for (const auto& pair : bigram_counts) {
            if (max_value < pair.second && pair.second > 1) {
                max_value = pair.second;
                key = pair.first;
                found_merge_condition = true;
            }
        }

        // SECOND: If we found a bigram to merge, do the merge operation
        if (found_merge_condition) {
            // Add the bigram to the vocabulary
            vocabulary.emplace(key);

            // Merge all instances of this bigram
            for (int i=0;i<string_vector.size()-1;i++) {
                // Iterate through the word vector
                if (key==string_vector[i]+string_vector[i+1]){
                    string_vector[i]=key;
                    string_vector.erase(string_vector.begin()+i+1);
                }
            }

            // THIRD: Recalculate frequencies for next iteration
            bigram_counts.clear();
            Calculate_Bigram_Frequency(string_vector, bigram_counts);
        }
        // If no merge-able bigrams found, the while loop will exit naturally
    }
}


void Print_PreTokens(const std::vector<std::string> &string_vector) {
    std::cout << "Tokenized sequence: ";
    for (const auto& token : string_vector) {
        std::cout << "[" << token << "] ";
    }
    std::cout << std::endl;
}

void Print_Bigram_Frequencies(std::unordered_map<std::string,int>& bigram_counts) {
    std::cout << "Bigram Frequencies:\n";
    for (const auto& entry : bigram_counts) {
        std::cout << "'" << entry.first << "' : " << entry.second << "\n";
    }
}

void Print_Vocabulary_Contents(const std::unordered_set<std::string>& vocabulary) {
    std::cout << "=== VOCABULARY CONTENTS ===" << std::endl;
    std::cout << "Total tokens: " << vocabulary.size() << std::endl;
    for (const auto& token : vocabulary) {
        std::cout << "\"" << token << "\" ";
    }
    std::cout << std::endl << std::endl;
}

void Print_Final_Tokenized_Corpus(const std::vector<std::string> &string_vector) {
    for (const auto & i : string_vector) {
        std::cout << "Word : " << i << std::endl;
    }
}

//Freeze final vocabulary and assign token IDs
void Assign_Token_IDs(std::unordered_map<std::string,int>& tokenized_vocabulary,
                      const std::unordered_set<std::string>& vocabulary) {
    int token_id = 0;

    // Add special tokens with reserved IDs. Pad should be 0;
    tokenized_vocabulary["<PAD>"] = token_id++;

    // Assign sequential IDs to vocabulary tokens
    for (const auto& token : vocabulary) {
        tokenized_vocabulary[token] = token_id++;
    }

    // Add special tokens with reserved IDs
    tokenized_vocabulary["<UNK>"] = token_id++;  // Unknown token for unseen words. This is NOT POSSIBLE during training but only during inference.
    tokenized_vocabulary["<BOS>"] = token_id++;  // Beginning of sequence
    tokenized_vocabulary["<EOS>"] = token_id++;  // End of sequence
}

//given a sequence,it will tokenize it to integer IDs.
void Tokenize_String(std::vector<int>&tokens,std::unordered_map<std::string,int>& tokenized_vocabulary,std::vector<std::string>& string_vector){
    int token_id;
    for (const auto& token : string_vector) {
        // Convert token to ID, use UNK if not found

        auto it = tokenized_vocabulary.find(token);
        if (it != tokenized_vocabulary.end()) {
            token_id = it->second;
        } else {
            token_id = tokenized_vocabulary.at("<UNK>");
        }
        tokens.emplace_back(token_id);
    }
}

// Function to create batched sequences with fixed length using our Matrix library
Matrix Create_Fixed_Length_Batches(
        const std::vector<int>& all_tokens,
        int max_sequence_length
) {
    int sequences_size = all_tokens.size();

    if(sequences_size < max_sequence_length){
        throw std::invalid_argument("Sequence size must be at least 1 multiple of max sequence length");
    }

    float actual_number_of_batches = static_cast<float>(sequences_size) / max_sequence_length;
    int confirmed_number_of_batches = static_cast<int>(std::floor(actual_number_of_batches));
    int max_number_of_batches = static_cast<int>(std::ceil(actual_number_of_batches));

    // Each row of the matrix contains a max sequence length
    Matrix batch_token_matrix(max_number_of_batches, max_sequence_length);
    int token_index = 0;

    // Complete the cycle for the confirmed batches
    for (int i = 0; i < confirmed_number_of_batches; i++){
        for(int j = 0; j < max_sequence_length; j++){
            batch_token_matrix(i, j) = static_cast<float>(all_tokens[token_index++]);
        }
    }

    // Handle the final partial batch if it exists
    if (confirmed_number_of_batches < max_number_of_batches) {
        int remaining_tokens = sequences_size - token_index;

        for(int i = 0; i < max_sequence_length; i++){
            if(i < remaining_tokens){
                // Use remaining real tokens
                batch_token_matrix(confirmed_number_of_batches, i) = static_cast<float>(all_tokens[token_index++]);
            } else {
                // Pad the rest with zeros
                batch_token_matrix(confirmed_number_of_batches, i) = 0.0f;
            }
        }
    }

    return batch_token_matrix;
}

Matrix Batch_Tokenization_Pipeline(const std::string& input_text,int max_sequence_length,int max_vocabulary_size ) {

    // Step 1: Convert input text to pretokens (character-level + word boundaries)
    std::vector<std::string> string_vector;
    Word_To_PreTokens(string_vector, input_text);

    // Step 2: Create initial vocabulary from character-level tokens
    std::unordered_set<std::string> vocabulary;
    Create_Vocabulary(string_vector, vocabulary);

    // Step 3: Calculate initial bigram frequencies
    std::unordered_map<std::string, int> bigram_counts;
    Calculate_Bigram_Frequency(string_vector, bigram_counts);

    // Step 4: Perform greedy bigram merging to build domain-specific vocabulary
    Greedy_Bigram_Merging(string_vector, vocabulary, bigram_counts, max_vocabulary_size);

    // Step 5: Create token-to-ID mapping with special tokens
    std::unordered_map<std::string, int> tokenized_vocabulary;
    Assign_Token_IDs(tokenized_vocabulary, vocabulary);

    // Step 6: Convert string tokens to integer IDs
    std::vector<int> token_ids;
    Tokenize_String(token_ids, tokenized_vocabulary, string_vector);

    // Step 7: Create fixed-length batches for training
    Matrix batched_tokens = Create_Fixed_Length_Batches(token_ids, max_sequence_length);

    Print_Vocabulary_Contents(vocabulary);
    Print_Final_Tokenized_Corpus(string_vector);

    return batched_tokens;
}
