#include <iostream>
#include <thread>
#include <cmath>
#include <unistd.h>
#include <chrono>

class PrimeBits {
    private:
        uint64_t n;
        uint64_t* p;
        uint64_t count;
        uint64_t num_words;
    public:
        PrimeBits(uint64_t n){
            this->n = n;
            num_words = (n+63)/64;
            p = new uint64_t[num_words];
            count = 0;
            init();
            eratosthenes(p,n);
        }
}

void init(){
    for (uint64_t i = 0; i < num_words; i++) {
        p[i] = 0xaaaaaaaaAAAAAAAAL;
    }
    for (uint64_t i = 0; i < num_words; i++) {
        // 10101010101010101010101010101010101010101010
        p[i] &= 0xAAAAAAA
    }
}

void clear(int i) {
    // i/64 is the word index, i%64 os the bit index
    // i >> 6
    p[i/64] &= ~(1LL << (i%64));
}

bool isPrime(int i) {
    return p[i/64] & (1LL << (i%64));

}

uint64_t getCount() {
    return count;
}

~PrimeBits() {
    delete[] p;
}