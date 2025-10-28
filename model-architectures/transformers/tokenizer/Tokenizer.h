//
// Created by rakib on 18/6/2025.
// Updated on 24-25/07/2025


#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKENIZER_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKENIZER_H

#include <iomanip>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <vector>


class Tensor;

Tensor Tokenize_Data(const std::string& file_path,std::unordered_set<std::string>& vocabulary,std::unordered_map<std::string,
    int>& tokenized_vocabulary, int max_vocabulary_size, int desired_sequence_length, const bool verbose);
void Print_PreTokens(const std::vector<std::string> &string_vector);
void Print_Bigram_Frequencies(std::unordered_map<std::string,int>& bigram_counts);
void Print_Vocabulary_Contents(const std::unordered_set<std::string>& vocabulary);
void Print_Final_Tokenized_Corpus(const std::vector<std::string> &string_vector);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKENIZER_H
