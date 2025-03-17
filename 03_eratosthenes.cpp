#include <iostream>
#include <thread>
#include <cmath>
#include <unistd.h>
#include <chrono>

//2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17
//0  0  1  0  1  0  1  1  1   0   1   0   1   1   1   0

void eratosthenes(bool primes[], uint64_t n) {
    uint64_t count = 1;
    for (uint64_t i = 1; i <= n; i+=2) {
        primes[i] = true;
    }
    
    for (uint64_t i = 3; i <= n; i+=2) {
        if (primes[i]){
            count++;
            for (uint64_t j = i*i; j <= n; j += 2*i) {
                primes[j] = false;
            }
        }
    }

    for (uint64_t i = sqrt(n) + 1; i <= n; i++){
        if (primes[i]){
            count++;
        }
    }
}


int main() {
    uint64_t n = 100;
    bool primes[n+1];
    eratosthenes(primes, n);
    for (uint64_t i = 0; i <= n; i++) {
        if (primes[i]) {
            printf("%ld ", i);
        }
    }
}

//isprime = new bool[n+1]
//first calculate up to sqrt(n)
//multithreaded sieve
//example n = 10^9
//first calculate up to sqrt(n) = 33k
//divide up sqrt(n)....n into num_threads pieces
//a = sqrt(n) b = n; 
//thread 1: a, a + (b-a)/4
//thread 2: a + (b-a)/4 + 1, a + (b-a)/2
//thread 3: a + (b-a)/2 + 1, a + (b-a)*3/4
//thread 4: a + (b-a)*3/4, b 

//n = 30 example
//single threaded n=5
//remaining 6..30 (divide into 4 pieces, each thread does one piece)
//6..12 13..18 19..24 25..30