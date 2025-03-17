// 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
// 0  0  1  0  1  0  1  1  1   0   1   0   1   1   1   0                               1
// i=97
//O(log log n)
#include <iostream>
#include <cmath>

void eratosthenes(bool primes[], uint64_t n) {
    uint64_t count = 1;
    for (uint64_t i = 3; i <= n; i+= 2) {
        primes[i] = true;
    }

    for (uint64_t i = 3; i <= sqrt(n); i+= 2) {
        if (primes[i]) {
            count++;
            for (uint64_t j = i * i; j <= n; j += 2*i) {
                primes[j] = false;
            }
        }w
    }
    for (uint64_t i = sqrt(n) + 1; i <= n; i++) {
        if (primes[i]) {
            count++;
        }
    }
//    *pcount = count;
}

int main() {
    uint64_t n = 100;
    bool primes[n+1];
    eratosthenes(primes, n);
    for (uint64_t i = 0; i <= n; i++) {
        if (primes[i]) {
            printf("%d ", i);
        }
    }
}