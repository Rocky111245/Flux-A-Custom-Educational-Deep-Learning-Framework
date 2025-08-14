//
// Created by rakib on 18/6/2025.
// Updated on 24-26/07/2025
// Updated on 13/08/2025

#include "Tokenizer.h"
#include "Tensor_Library/Tensor_Library.h"


/*
 Implementation of Byte-Pair Encoding Algorithm. In this algorithm implementation,contrary to other code in this
 project, special attention was given to the time complexity and overall software optimization techniques enabled for the
 BPE Merging, which might reduce readability.
 This is because,in a previous test, the time complexity was so high that respectable data size was taking too long
 to tokenize.
// Updated on 13/08/2025- BPE is fast enough to tokenize
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
//One 2D Tensor stores all tokenized sequences into batches
Tensor Tensorize_Token_Vector(const std::vector<int>& tokens, const std::unordered_map<std::string,int>& tokenized_vocabulary, const int& desired_sequence_length) {
    const int content_len    = desired_sequence_length - 2;   // real tokens between SOS/EOS
    const int pad_id         = tokenized_vocabulary.at("<PAD>");
    const int sos_id         = tokenized_vocabulary.at("<BOS>");
    const int eos_id         = tokenized_vocabulary.at("<EOS>");

    // we find the ACTUAL sequences (rows) of tensors we require to accommodate all tokens
    const float actual_sequence_number =
        static_cast<float>(tokens.size()) / static_cast<float>(content_len); // keep float precision
    const int tensor_rows = std::ceil(actual_sequence_number);
    const int full_sequence_number = static_cast<int>(std::floor(actual_sequence_number));
    //----------------------------------------------------------------

    // shape: (rows, desired_sequence_length, 1)
    Tensor all_batches(tensor_rows, desired_sequence_length, 1);

    int token_vector_index = 0;

    // fill complete rows
    for (int i = 0; i < full_sequence_number; ++i) {
        all_batches(i, 0, 0) = sos_id;
        for (int j = 1; j <= content_len; ++j)          // inclusive upper bound
        {
            all_batches(i, j, 0) = tokens[token_vector_index++];
        }
        all_batches(i, desired_sequence_length - 1, 0) = eos_id;
    }

    // final partial row
    if (token_vector_index < static_cast<int>(tokens.size())) {
        const int row = full_sequence_number;
        all_batches(row, 0, 0) = sos_id;
        int col = 1;
        while (token_vector_index < static_cast<int>(tokens.size())) {
            all_batches(row, col++, 0) = tokens[token_vector_index++];
        }
        all_batches(row, col, 0) = eos_id;              // place EOS immediately after last token

        // pad remainder
        for (++col; col < desired_sequence_length - 1; ++col) {
            all_batches(row, col, 0) = pad_id;
        }
    }

    return { all_batches };
}


//main driver code. Please call file path as (Load file std::ifstream file("../Transformer_Architecture/Test_Data/test_for_tokenizer.txt");
//Study Tokenizer.cpp file to understand implementation.
Tensor Tokenize_Data(const std::string &file_path, std::unordered_set<std::string> &vocabulary, std::unordered_map<std::string, int>& tokenized_vocabulary, int max_vocabulary_size, int desired_sequence_length, const bool verbose)
{
    if (verbose) std::cout << "=== BPE TOKENIZER RUNNING ===\n";

    std::ifstream file(file_path);
    if (!file)
        throw std::runtime_error("Cannot open file: " + file_path);

    const std::string file_content((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());

    if (verbose)
        std::cout << "Loaded " << file_content.size() << " characters\n";

    // 1) Pretokenise
    std::vector<std::string> pretokens;
    Word_To_PreTokens(pretokens, file_content);

    // 2) Build / reuse caller-supplied vocabulary
    Create_Vocabulary(pretokens, vocabulary);

    // 3) Merge
    std::unordered_map<std::string, int> bigram_counts;
    Calculate_Bigram_Frequency(pretokens, bigram_counts);

    if (verbose) std::cout << "Starting merging operations\n";
    Greedy_Bigram_Merging(pretokens, vocabulary, bigram_counts, max_vocabulary_size);
    if (verbose) std::cout << "Ended merging operations\n";

    // 4) Build caller-supplied token→ID map
    tokenized_vocabulary.clear();

    Assign_Token_IDs(tokenized_vocabulary, vocabulary);
    if (verbose) std::cout << "Ended Assign_Token_IDs\n";

    // 5) Convert to IDs
    std::vector<int> token_ids;
    Tokenize_String(token_ids, tokenized_vocabulary, pretokens);
    if (verbose) std::cout << "Ended Tokenize_String\n";

    Tensor tokenized_tensor=Tensorize_Token_Vector(token_ids,tokenized_vocabulary,desired_sequence_length);
    if (verbose) std::cout << "Ended tokenized_tensor\n";

    if (verbose) Print_Vocabulary_Contents(vocabulary);
    return tokenized_tensor;
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
        std::cout << token <<std::endl;;
    }
    std::cout << std::endl << std::endl;
}

void Print_Final_Tokenized_Corpus(const std::vector<std::string> &string_vector) {
    std::cout << "=== FINAL CORPUS ===" << std::endl;
    for (const auto & i : string_vector) {
        std::cout << "Word : " << i << std::endl;
    }
}