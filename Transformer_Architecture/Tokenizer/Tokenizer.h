//
// Created by rakib on 18/6/2025.
//


#ifndef _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKENIZER_H
#define _DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKENIZER_H

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <string>
#include <cctype>
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <array>

class Matrix;

Matrix Batch_Tokenization_Pipeline(const std::string& input_text,
                                   int max_sequence_length = 256,
                                   int max_vocabulary_size = 2000);
void Print_PreTokens(const std::vector<std::string> &string_vector);
void Print_Bigram_Frequencies(std::unordered_map<std::string,int>& bigram_counts);
void Print_Vocabulary_Contents(const std::unordered_set<std::string>& vocabulary);
void Print_Final_Tokenized_Corpus(const std::vector<std::string> &string_vector);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKENIZER_H
