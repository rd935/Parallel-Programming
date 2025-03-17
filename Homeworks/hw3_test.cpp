#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <filesystem>

using namespace std;

struct word_info {
    uint64_t count; // total number of times the word was found
    uint32_t in_books; // in how many books
    int32_t last_book; // number of last book it was found in
};

class Dict {
private:
    unordered_map<string, word_info> dict;
public:
    Dict() {}

    void add_word(const string& word, int book) {
        if (dict.find(word) == dict.end()) {
            dict[word] = { 1, 1, book }; // First time seeing this word
        } else {
            dict[word].count++;
            if (book > dict[word].last_book) {
                dict[word].last_book = book; // Update last book
            }
            // Update in_books if this is a new book
            if (book > dict[word].in_books) {
                dict[word].in_books++;
            }
        }
    }

    void print_summary() const {
        cout << "Summary of word counts:\n";
        for (const auto& entry : dict) {
            const string& word = entry.first;
            const word_info& info = entry.second;
            cout << "Word: \"" << word << "\", Count: " << info.count 
                 << ", Distinct Books: " << info.in_books 
                 << ", Last Book: " << info.last_book << endl;
        }
    }
};

// Function to clean words by removing punctuation
void clean_word(string& word) {
    word.erase(remove_if(word.begin(), word.end(), [](unsigned char c) {
        return ispunct(c); // Remove punctuation
    }), word.end());
}

// Open a single book
Dict d;

void openfile(const std::filesystem::path& path, int book_num) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << path << std::endl;
        return;
    }

    string word;
    while (file >> word) {
        clean_word(word);
        transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return std::tolower(c); });
        if (!word.empty()) { // Ensure the word is not empty after cleaning
            d.add_word(word, book_num);
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <directory_path>" << std::endl;
        return 1;
    }

    string path = argv[1];
    int book_num = 0;

    try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                std::cout << "Found .txt file: " << entry.path().filename() << std::endl;
                openfile(entry.path(), ++book_num);
            }
        }
    } catch (const std::filesystem::filesystem_error& err) {
        std::cerr << "Filesystem error: " << err.what() << std::endl;
    }

    d.print_summary(); // Print the summary of words and their counts

    return 0;
}
