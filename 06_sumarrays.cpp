#include <iostream>
#include <chrono>
#include <immintrin.h>
using namespace std;

void benchmark(string name, uint64_t (*f)(uint64_t *, int) , uint64_t *a, int n){
    auto t0 = chrono::high_resolution_clock().now();
    f(a,n);
    auto t1 = chrono::high_resolution_clock().now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(t1-t0).count();
    cout << name << "elapsed = " << elapsed << "ms\n";
}

uint64_t scan_array(uint64_t a[], int n){
    for (int i = 0; i < n; i++){
        a[i];
    }

    return 0;
}

uint64_t sum_array(uint64_t a[], int n){
    uint64_t sum = 0;
    for (int i = 0; i < n; i++){
        sum += a[i];
    }

    return 0;
}

uint64_t sum_array2(uint64_t a[], int n){
    uint64_t sum = 0;
    for (int i = 0; i < n; i++){
        sum += a[i];
    }

    return sum;
}

// __m256i sum_array3(__m256i a[], int n){
//     __m256i sum = _mm256_setzero_si256(); //ymm0
//     for (int i = 0; i < n; i+=8){
//         __m256i x = _mm256_loadu_si256(&a[i]); //ymm1?
//         sum = _mm256_add_epi32(sum, x); //ymm0
//         //sum = (sum1, sum2, sum2, ... sum8)
//         //add all the components of sum togeter
        
//         //sum += a[i];
//     }

//     return sum;
// }

int main(){
    const int n = 1'000'000'000;
    uint64_t *a = new uint64_t[n];
    //__m256i a = aligned_alloc<uint64_t>[n];
    benchmark("Scan array: ", scan_array, a, n);
    benchmark("Sum array: ", sum_array, a, n);
    benchmark("Sum array 2: ", sum_array2, a, n);
    //sum_array3(a, n);

    return 0;
}