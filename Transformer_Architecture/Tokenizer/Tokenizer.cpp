//
// Created by rakib on 18/6/2025.
// Updated on 24-26/07/2025

#include "Tokenizer.h"
#include "MatrixLibrary.h"


/*
 Implementation of Byte-Pair Encoding Algorithm. In this algorithm implementation,contrary to other code in this
 project, special attention was given to the time complexity and C++ and overall software optimization techniques enabled.
 This is because,in a previous test, the time complexity was so high that respectable data size was taking too long
 to tokenize.
*/


//This converts the corpus to a vector of strings having single characters every index
//This is because later they will be merged, and we will ultimately need a string vector
void Word_To_PreTokens(std::vector<std::string> &string_vector, const std::string& sentence) {
    const size_t sentence_size = sentence.length();
    string_vector.clear();
    // Pre-allocate memory: estimate ~2x input size. More than enough.
    string_vector.reserve(sentence_size*2);

    // Add initial boundary marker only if sentence doesn't start with whitespace
    if (!sentence.empty() && !std::isspace(sentence[0])) {
        string_vector.emplace_back("_");
    }
    // Check previous character in sentence
    for (size_t i = 0; i < sentence_size; ++i) {
        if (std::isspace(sentence[i])) {
            // Only add boundary token if previous character wasn't also whitespace. These checks are needed to prevent
            // vocabulary pollution
            if (i == 0 || !std::isspace(sentence[i-1])) {
                string_vector.emplace_back("_");
            }
        } else {
            string_vector.emplace_back(1, sentence[i]);
        }
    }
}


void Create_Vocabulary(const std::vector<std::string>&string_vector,std::unordered_set<std::string> &vocabulary){
    vocabulary.clear();
    // Reserve space
    vocabulary.reserve(string_vector.size());

    for (const auto& token : string_vector) {
        vocabulary.emplace(token);
    }
}



void Calculate_Bigram_Frequency(const std::vector<std::string>& string_vector,
                                std::unordered_map<std::string,int>& bigram_counts) {

    std::string bigram_string;
    bigram_string.clear();
    bigram_string.reserve(200); // reserve memory to prevent reallocations.

    for (size_t i = 0; i < string_vector.size() - 1; ++i) {
        bigram_string.clear(); // reuse the reserved memory
        bigram_string.append(string_vector[i]);
        bigram_string.append(string_vector[i + 1]);
        bigram_counts[bigram_string]++;
    }
}



void Greedy_Bigram_Merging(std::vector<std::string>& string_vector,
                           std::unordered_set<std::string>& vocabulary,
                           std::unordered_map<std::string,int>& bigram_counts,
                           int max_vocabulary_size) {

    bool found_merge_condition = true;
    int max_value = 0;
    std::string key;
    key.clear();
    key.reserve(200); // Reserve space for merged tokens

    std::string bigram_buffer;
    bigram_buffer.clear();
    bigram_buffer.reserve(200); // Reusable buffer for bigram construction

    std::vector<std::string> new_string_vector;
    // Reserve space for efficiency
    new_string_vector.clear();
    new_string_vector.reserve(string_vector.size());


    while (found_merge_condition && vocabulary.size() < max_vocabulary_size) {
        // Reset for this iteration
        found_merge_condition = false;
        max_value = 0;
        key.clear();

        new_string_vector.clear();
        // FIRST: Find the most frequent bigram
        for (const auto& pair : bigram_counts) {
            if (pair.second > max_value && pair.second >= 2) {
                max_value = pair.second;
                key = pair.first;
                found_merge_condition = true;
            }
        }

        // SECOND: If we found a bigram to merge, perform merge operation
        if (found_merge_condition) {
            // Add the merged bigram to vocabulary
            vocabulary.emplace(key);
            // Merge all instances of this bigram
            for (size_t i = 0; i < string_vector.size(); ) {
                // Check if we can form a bigram and if it matches our target
                if (i < string_vector.size() - 1) {
                    bigram_buffer.clear();
                    bigram_buffer.append(string_vector[i]);
                    bigram_buffer.append(string_vector[i + 1]);

                    if (key == bigram_buffer) {
                        // Merge: add the combined token and skip both elements
                        new_string_vector.emplace_back(key);
                        i += 2;
                        continue;
                    }
                }

                // No merge: add current token and advance by 1
                new_string_vector.emplace_back(string_vector[i]);
                i++;
            }

            // Apply merge result
            string_vector = std::move(new_string_vector);

            // THIRD: Recalculate frequencies for next iteration. Thats it,repeat this process until it hits max vocab
            //or no merge conditions found.
            bigram_counts.clear();
            Calculate_Bigram_Frequency(string_vector, bigram_counts);
        }
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
void Tokenize_String(std::vector<int>&tokens,std::unordered_map<std::string,int>& tokenized_vocabulary, const std::vector<std::string>& string_vector){
    int token_id;
    tokens.clear();
    tokens.reserve(string_vector.size()*2); //Just to make sure no reallocations are needed.
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
    std::cout << "=== FINAL CORPUS ===" << std::endl;
    for (const auto & i : string_vector) {
        std::cout << "Word : " << i << std::endl;
    }
}
