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



void Word_To_PreTokens(std::vector<std::string> &string_vector, const std::string& sentence);
void Create_Vocabulary(const std::vector<std::string>&string_vector,std::unordered_set<std::string> &vocabulary);
void Calculate_Bigram_Frequency(const std::vector<std::string>& string_vector,std::unordered_map<std::string,int>& bigram_counts) ;
void Assign_Token_IDs(std::unordered_map<std::string,int>& tokenized_vocabulary,
                      const std::unordered_set<std::string>& vocabulary);
void Greedy_Bigram_Merging(std::vector<std::string>& string_vector,
                           std::unordered_set<std::string>& vocabulary,
                           std::unordered_map<std::string,int>& bigram_counts,
                           int max_vocabulary_size);
void Tokenize_String(std::vector<int>&tokens,std::unordered_map<std::string,int>& tokenized_vocabulary, const std::vector<std::string>& string_vector);
void Print_PreTokens(const std::vector<std::string> &string_vector);
void Print_Bigram_Frequencies(std::unordered_map<std::string,int>& bigram_counts);
void Print_Vocabulary_Contents(const std::unordered_set<std::string>& vocabulary);
void Print_Final_Tokenized_Corpus(const std::vector<std::string> &string_vector);

#endif //_DISCRIMINATIVE_DENSE_NEURAL_NETWORK_FRAMEWORK_TOKENIZER_H
