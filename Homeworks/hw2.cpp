//Ritwika Das -- rd935
//g++ -g hw2.cpp -o hw2.exe --> to compile
//./hw2.exe <limit> <num_threads> --> input

#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <mutex>


class prime_bits {
public:
    std::vector<uint64_t> bits;
    uint64_t size;

    prime_bits(uint64_t n) : size(n) {
        bits.resize((n / 64) + 1, ~0ULL);  // Set all bits to 1 (assuming all are prime)
        bits[0] &= ~1ULL;                  // Mark 0 and 1 as non-prime
    }

    // Mark a number as non-prime
    void mark_non_prime(uint64_t n) {
        bits[n / 64] &= ~(1ULL << (n % 64));
    }

    // Check if a number is prime
    bool is_prime(uint64_t n) const {
        return bits[n / 64] & (1ULL << (n % 64));
    }

    // Eratosthenes Sieve up to sqrt(n) (bitwise)
    void eratosthenes(uint64_t n) {
        for (uint64_t p = 2; p * p <= n; p++) {
            if (is_prime(p)) {
                for (uint64_t multiple = p * p; multiple <= n; multiple += p) {
                    mark_non_prime(multiple);
                }
            }
        }
    }
};

// Thread function for parallel segmented
void parallel_sieve_segment(uint64_t start, uint64_t end, const prime_bits& small_primes, prime_bits& segment) {
    for (uint64_t p = 2; p * p <= end; p++) {
        if (small_primes.is_prime(p)) {
            uint64_t multiple = std::max(p * p, (start + p - 1) / p * p);  // Start from the correct multiple
            for (; multiple <= end; multiple += p) {
                segment.mark_non_prime(multiple - start);
            }
        }
    }
}

uint64_t parallel_eratosthenes(uint64_t n, uint64_t num_threads) {
    uint64_t sqrt_n = static_cast<uint64_t>(std::sqrt(n));

    // Step 1: Sieve for small primes up to sqrt(n)
    prime_bits small_primes(sqrt_n + 1);
    small_primes.eratosthenes(sqrt_n);

    // Step 2: Prepare segmented sieve in parallel
    uint64_t block_size = 32768;  // Example block size
    std::vector<std::thread> threads;
    std::mutex prime_count_mutex;
    uint64_t prime_count = 0;

    auto worker = [&](uint64_t start, uint64_t end) {
        prime_bits segment(end - start + 1);  // Bitwise segment for the range
        parallel_sieve_segment(start, end, small_primes, segment);

        // Count primes in this segment
        uint64_t local_prime_count = 0;
        for (uint64_t i = 0; i <= (end - start); ++i) {
            if (segment.is_prime(i)) local_prime_count++;
        }

        // Update the total prime count
        std::lock_guard<std::mutex> guard(prime_count_mutex);
        prime_count += local_prime_count;
    };

    // Step 3: Divide the range into blocks and assign threads
    uint64_t start = sqrt_n + 1;
    while (start <= n) {
        uint64_t end = std::min(start + block_size - 1, n);
        threads.clear(); // Clear the vector before using it

        // Create threads to handle different blocks
        for (uint64_t i = 0; i < num_threads; ++i) {
            uint64_t thread_start = start + i * block_size;
            uint64_t thread_end = std::min(start + (i + 1) * block_size - 1, n);
            if (thread_start > n) break; // If the start exceeds n, stop
            threads.emplace_back(worker, thread_start, thread_end);
        }

        // Wait for all threads to finish
        for (auto& thread : threads) {
            if (thread.joinable()) { // Check if thread is joinable before joining
                thread.join();
            }
        }

        // Move to the next set of blocks
        start += block_size * num_threads;
    }

    return prime_count;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << "<limit> <num_threads> \n";
        return 1;
    }

    uint64_t n = atol(argv[1]);
    uint64_t num_threads = atol(argv[2]);
    uint64_t total_primes = parallel_eratosthenes(n, num_threads);
    std::cout << "Total primes: " << total_primes << std::endl;
    return 0;



}
