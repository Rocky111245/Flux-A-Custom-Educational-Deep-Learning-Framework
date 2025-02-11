#include <string>
#include <iostream>
#include <cctype>  // for tolower, isspace, isalnum
#include <unordered_set>
#include <unordered_map>
#include <map>


void Convert_to_Lowercase(std::string& new_string, const std::string& original_string) {
    new_string.clear();  // Clear the string before adding new characters

    for (char item : original_string) {
        char lower_char = std::tolower(item);  // Store the lowercase version

        if (std::isspace(item) || std::isalnum(item)) {
            new_string.push_back(lower_char);
        }
    }
}

void Create_Vocabulary(std::unordered_set<char>& character_set, const std::string& s) {

    // Create character set (corpus)
    for (char c : s) {
        character_set.insert(c);
    }

    // Create bigram map
    std::map<std::pair<char,char>, int> token_map;


    for (size_t i = 0; i < s.size() - 1; i++) {
        token_map[{s[i], s[i+1]}]++;
    }
}





void Print_String(const std::string& string) {
    for (char item : string) {
        std::cout << item;
    }
    std::cout << std::endl;  // Add newline at end
}

int main() {
    std::string example = "I love swimming.";
    std::string new_string;  // No need to initialize with example

    Convert_to_Lowercase(new_string, example);
    Print_String(new_string);

    return 0;
}