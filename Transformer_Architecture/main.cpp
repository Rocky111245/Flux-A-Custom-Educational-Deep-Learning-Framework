#include <iostream>
#include <fstream>

#include "Tensor_Library/Tensor_Library.h"
#include "Tokenizer/Tokenizer.h"


// Updated on 24-26/07/2025



//Right now testing the updated tokenizer (26-07-2025)
int main() {
    std::cout << "=== BPE TOKENIZER TESTING ===" << std::endl;

    try {

        // Step 1: Convert input text to pretokens (character-level + word boundaries)
        std::vector<std::string> string_vector;

        // Step 2: Create initial vocabulary from character-level tokens
        std::unordered_set<std::string> vocabulary;

        // Step 3: Calculate initial bigram frequencies
        std::unordered_map<std::string, int> bigram_counts;

        // Step 4: Perform greedy bigram merging to build domain-specific vocabulary
        // Step 5: Create token-to-ID mapping with special tokens
        std::unordered_map<std::string, int> tokenized_vocabulary;

        // Step 6: Convert string tokens to integer IDs
        std::vector<int> token_ids;
        Tensor tensor;

        tensor=Tokenize_Data("../Transformer_Architecture/Test_Data/long_string_tokenizer_text.txt",vocabulary,tokenized_vocabulary,3000,312,true);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}