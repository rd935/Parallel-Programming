//Ritwika Das -- rd935
//g++ -std=c++17 -pthread hw3.cpp -o hw3 --> to compile
//./hw3 <directory_path> <num_threads> --> input


#include <iostream>
#include <fstream>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <filesystem>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <condition_variable>

using namespace std;

struct word_info {
    uint64_t count; // total number of times the word was found
    uint32_t in_books; // in how many books
    int32_t last_book; // number of last book it was found in
};

class Dict {
private:
    unordered_map<string, word_info> dict;
    mutex mtx; // Mutex for synchronization
public:
    Dict() {}

    void add_word(const string& word, int book) {
        lock_guard<mutex> lock(mtx); // Lock the mutex for thread safety
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
        cout << "Summary of word counts (only words with more than 1 occurrence):\n";
        for (const auto& entry : dict) {
            const string& word = entry.first;
            const word_info& info = entry.second;
            if (info.count > 1) { // Only print words with count greater than 1
                cout << "Word: \"" << word << "\", Count: " << info.count 
                     << ", Distinct Books: " << info.in_books 
                     << ", Last Book: " << info.last_book << endl;
            }
        }
    }
};

// Function to clean words by removing punctuation
void clean_word(string& word) {
    word.erase(remove_if(word.begin(), word.end(), [](unsigned char c) {
        return ispunct(c); // Remove punctuation
    }), word.end());
}

// Worker function to process files
void process_files(queue<filesystem::path>& files, Dict& dict, mutex& queue_mutex, condition_variable& cond_var, bool& done, int book_num) {
    while (true) {
        filesystem::path file_path;

        {
            unique_lock<mutex> lock(queue_mutex);
            cond_var.wait(lock, [&] { return !files.empty() || done; }); // Wait for files or done signal

            // Check exit condition
            if (files.empty() && done) {
                cout << "Thread " << book_num << " exiting: No more files to process." << endl;
                break; // Exit if done and queue is empty
            }

            // If there are files, get the next file path
            if (!files.empty()) {
                file_path = files.front();
                files.pop();
            }
        } // Release the lock here before processing the file

        // Process the file if file_path is valid
        if (!file_path.empty()) {
            ifstream file(file_path);
            if (!file.is_open()) {
                cerr << "Thread " << book_num << ": Failed to open file: " << file_path << endl;
                continue;
            }

            string word;
            while (file >> word) {
                clean_word(word);
                transform(word.begin(), word.end(), word.begin(), [](unsigned char c) { return tolower(c); });
                if (!word.empty()) {
                    dict.add_word(word, book_num); // Use book_num to track the file
                }
            }

            cout << "Thread " << book_num << ": Finished processing file: " << file_path.filename() << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <directory_path> <num_threads>" << endl;
        return 1;
    }

    string path = argv[1];
    int num_threads = stoi(argv[2]);

    // Validate thread count
    if (num_threads < 2 || num_threads > 4) {
        cerr << "Number of threads must be between 2 and 4." << endl;
        return 1;
    }

    Dict d;
    queue<filesystem::path> files;
    mutex queue_mutex;
    condition_variable cond_var;
    bool done = false;

    // Collect all text files
    try {
        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                files.push(entry.path());
                cout << "Found .txt file: " << entry.path().filename() << endl;
            }
        }
    } catch (const std::filesystem::filesystem_error& err) {
        cerr << "Filesystem error: " << err.what() << endl;
        return 1;
    }

    // Start worker threads
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(process_files, ref(files), ref(d), ref(queue_mutex), ref(cond_var), ref(done), i);
    }

    // Notify all threads that files are available
    {
        lock_guard<mutex> lock(queue_mutex);
        done = false; // Set done to false before notifying
        cond_var.notify_all(); // Notify all threads
    }

    // Wait for all files to be processed
    {
        unique_lock<mutex> lock(queue_mutex);
        done = true; // Set done to true
        cond_var.notify_all(); // Notify all threads to exit if waiting
    }

    // Join all threads
    for (auto& t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    // Print the summary after all threads have completed
    d.print_summary(); // Print the summary of words and their counts

    return 0;
}
