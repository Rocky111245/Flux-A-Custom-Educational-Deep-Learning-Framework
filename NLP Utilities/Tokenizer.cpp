#include <string>
#include <iostream>
#include <cctype>  // for tolower, isspace, isalnum
#include <unordered_set>
#include <unordered_map>
#include <map>
#include <vector>
#include <array>


//Implementation of Byte-Pair Encoding Algorithm



void Word_To_PreTokens(std::unordered_map<int, std::vector<std::string>>& string_map, std::string& sentence) {
    int string_size = sentence.length();
    string_map[0].emplace_back("_"); // Add start marker for first word
    int index = 0;

    for (int i = 0; i < string_size; i++) {
        if (std::isspace(sentence[i])) {
            index++;
            string_map[index].emplace_back("_"); // New word start
        }
        else {
            string_map[index].emplace_back(1, sentence[i]);
        }
    }
}


void Create_Vocabulary(std::unordered_map<int, std::vector<std::string>>& string_map,std::unordered_set<std::string> &vocabulary){
    //this gives us the bulk size (how many keys)
    int size_of_character_map=string_map.size();

    for(int i=0;i<size_of_character_map;i++){
        for(int j=0;j<string_map[i].size();j++){
            vocabulary.emplace(string_map[i][j]);
        }
    }

}



void Create_Bigram_Frequency(std::unordered_map<int, std::vector<std::string>>& string_map,
                             std::unordered_map<std::string,int>& bigram_counts) {

    for(const auto& word_entry : string_map) {
        const std::vector<std::string>& tokens = word_entry.second;

        for(size_t j = 0; j + 1 < tokens.size(); j++) {
            std::string bigram_string = tokens[j] + tokens[j+1];
            bigram_counts[bigram_string]++;
        }
    }
}
void Merge_Bigrams(std::unordered_map<int, std::vector<std::string>>& string_map,
                   std::unordered_set<std::string> &vocabulary,
                   std::unordered_map<std::string,int>& bigram_counts) {

    bool found_merge_condition = true;
    int max_value = 0;
    std::string key;
    std::string bigram_string;

    while (found_merge_condition) {
        // Reset for this iteration
        found_merge_condition = false;
        max_value = 0;
        key.clear();

        // FIRST: Find the most frequent bigram (complete this search)
        for (const auto& pair : bigram_counts) {
            if (max_value < pair.second && pair.second > 1) {
                max_value = pair.second;
                key = pair.first;
                found_merge_condition = true;
            }
        }

        // SECOND: If we found a bigram to merge, do the merge operation
        if (found_merge_condition) {
            vocabulary.emplace(key);

            // Merge all instances of this bigram
            for (auto& word_pair : string_map) {
                std::vector<std::string>& current_word = word_pair.second;

                // Iterate through the word vector
                // Use reverse iteration to avoid index shifting complexity
                for (int k = static_cast<int>(current_word.size()) - 2; k >= 0; k--) {
                    if (current_word[k] + current_word[k + 1] == key) {
                        current_word[k] = current_word[k] + current_word[k + 1];
                        current_word.erase(current_word.begin() + k + 1);
                    }
                }
            }

            // THIRD: Recalculate frequencies for next iteration
            bigram_counts.clear();
            Create_Bigram_Frequency(string_map, bigram_counts);
        }
        // If no mergeable bigrams found, the while loop will exit naturally
    }
}









int main() {
    std::string example = "abab abab";
    std::unordered_map<int, std::vector<std::string>> string_map;
    std::unordered_set<std::string> vocabulary;
    std::unordered_map<std::string, int> bigram_counts;
    Word_To_PreTokens(string_map, example);
    Create_Vocabulary(string_map, vocabulary);
    Create_Bigram_Frequency(string_map, bigram_counts);
    Merge_Bigrams(string_map,vocabulary,bigram_counts);
    return 0;
}