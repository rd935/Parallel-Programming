//Author: Ritwika Das, rd935

#include <omp.h>
#include <immintrin.h> 
#include <iostream>
#include <chrono>

//Compile: g++ -fopenmp -mavx2 -mfma -o hw5 hw5.cpp
//Output: ./hw5

// Threads: 1, Time: 1.38138 seconds
// Threads: 2, Time: 0.763224 seconds
// Threads: 4, Time: 0.347521 seconds
// Threads: 8, Time: 0.274542 seconds
// Threads: 24, Time: 0.226482 seconds

void multiplyMatrixVectorized(float a[], float b[], float c[], int n, int threads) {
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            __m256 dotProduct = _mm256_setzero_ps(); 
            for (int k = 0; k < n; k += 8) {

                __m256 A = _mm256_loadu_ps(&a[i*n + k]); 
                __m256 B = _mm256_loadu_ps(&b[k*n + j]); 
                
                dotProduct = _mm256_fmadd_ps(A, B, dotProduct);
            }

            float result[8];
            _mm256_storeu_ps(result, dotProduct);
            c[i*n + j] = result[0] + result[1] + result[2] + result[3] +
                         result[4] + result[5] + result[6] + result[7];
        }
    }
}

void benchmarkMatrixVectorized(float a[], float b[], float c[], int n, int threads) {
    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrixVectorized(a, b, c, n, threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Threads: " << threads << ", Time: " << duration.count() << " seconds\n";
}

int main() {
    int n = 1024;  // Matrix size
    int threadsArray[] = {1, 2, 4, 8, 24};  // Number of threads to use

    // Allocate memory for matrices
    float* a = new float[n*n];
    float* b = new float[n*n];
    float* c = new float[n*n];

    // Initialize matrices (identity matrices for testing)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            a[i*n + j] = (i == j) ? 1.0f : 0.0f;
            b[i*n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Perform matrix multiplication with vectorization and OpenMP
    for (int threads : threadsArray) {
        benchmarkMatrixVectorized(a, b, c, n, threads);
    }

    // Clean up memory
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}
