//Ritwika Das - rd935
//Eratosthenes Homework

#include <iostream>
#include <thread>
#include <cmath>
#include <unistd.h>
#include <vector>
#include <chrono>
#include <atomic>

void eratosthenes(std::vector<bool>& primes, uint64_t a, uint64_t b, uint64_t n) {

    for (uint64_t i = 2; i*i <= n; i++) {
        if (primes[i]){
            uint64_t first = std::max(i*i, (a + i - 1)/i*i);
            for (uint64_t j = i*i; j <= b; j += i) {
                primes[j] = false;
            }
        }
    }
}

void multithreads(uint64_t n, int numThreads) {
    std::vector<bool> primes(n + 1, true);
    primes[0] = primes[1] = false;

    std::vector<std::thread> threads;
    uint64_t chunkSize = n/numThreads;

    for (int i = 0; i < numThreads; i++) {
        uint64_t start = i*chunkSize + 2;
        uint64_t end = (i == numThreads - 1) ? n : (i + 1)*chunkSize + 1;
        threads.emplace_back(eratosthenes, std::ref(primes), start, end, n);
    }

    for (auto& t : threads) {
        t.join();
    }

    uint64_t count = 0;
    for(uint64_t i = 2; i <= n; i++){
        if (primes[i]) {
            count++;
            //std::cout << i << " ";
        }
    }

    std::cout << '\n' << "Total primes: " << count << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "<limit> <num_threads> \n";
        return 1;
    }
    uint64_t n = atol(argv[1]);
    int numThreads = atoi(argv[2]);

    multithreads(n, numThreads);
    
    return 0;
}