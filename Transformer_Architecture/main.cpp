#include <iostream>
#include <fstream>
#include "Tokenizer/Tokenizer.h"
#include "Token_Embedding/Token_Embedding.h"

// Updated on 24-26/07/2025



//Right now testing the updated tokenizer (26-07-2025)
int main() {
    std::cout << "=== BPE TOKENIZER TESTING ===" << std::endl;

    try {
        // Load file
        std::ifstream file("../Transformer_Architecture/Test_Data/test_for_tokenizer.txt");

        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file");
        }

        std::string file_content((std::istreambuf_iterator<char>(file)),
                                 std::istreambuf_iterator<char>());

        std::cout << "Loaded " << file_content.size() << " characters from file" << std::endl;

        // Step 1: Convert input text to pretokens (character-level + word boundaries)
        std::vector<std::string> string_vector;
        Word_To_PreTokens(string_vector, file_content);

        // Step 2: Create initial vocabulary from character-level tokens
        std::unordered_set<std::string> vocabulary;
        Create_Vocabulary(string_vector, vocabulary);

        // Step 3: Calculate initial bigram frequencies
        std::unordered_map<std::string, int> bigram_counts;
        Calculate_Bigram_Frequency(string_vector, bigram_counts);

        // Step 4: Perform greedy bigram merging to build domain-specific vocabulary
        std::cout << "Starting merging operations" << std::endl;
        Greedy_Bigram_Merging(string_vector, vocabulary, bigram_counts, 3000);
        std::cout << "Ended merging operations" << std::endl;

        // Step 5: Create token-to-ID mapping with special tokens
        std::unordered_map<std::string, int> tokenized_vocabulary;
        Assign_Token_IDs(tokenized_vocabulary, vocabulary);

        // Step 6: Convert string tokens to integer IDs
        std::vector<int> token_ids;
        Tokenize_String(token_ids, tokenized_vocabulary, string_vector);

        std::cout << "BIGRAM FREQUENCY" << std::endl;
        Print_Bigram_Frequencies(bigram_counts);
        std::cout << "VOCABULARY" << std::endl;
        Print_Vocabulary_Contents(vocabulary);
        std::cout << "FINAL CORPUS TO BE TOKENIZED" << std::endl;
        Print_Final_Tokenized_Corpus(string_vector);

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }


    return 0;
}