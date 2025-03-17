//Author: Ritwika Das, rd935

#include <omp.h>
#include <iostream>
#include <chrono>
#include <immintrin.h>

//Compile: g++ -O2 -fopenmp -mavx2 -mfma -o hw4 hw4.cpp
//Output: ./hw4

//Outputs:
// Threads: 1, Time: 6.01146 seconds
// Threads: 2, Time: 2.99939 seconds
// Threads: 4, Time: 1.51027 seconds
// Threads: 8, Time: 0.802937 seconds
// Threads: 24, Time: 0.620161 seconds
// With Transposed B - Threads: 1, Time: 1.98541 seconds
// With Transposed B - Threads: 2, Time: 1.17215 seconds
// With Transposed B - Threads: 4, Time: 0.619984 seconds
// With Transposed B - Threads: 8, Time: 0.451634 seconds
// With Transposed B - Threads: 24, Time: 0.319941 seconds
// Vectorized Threads: 1, Time: 0.422754 seconds
// Vectorized Threads: 2, Time: 0.20675 seconds
// Vectorized Threads: 4, Time: 0.144601 seconds
// Vectorized Threads: 8, Time: 0.117394 seconds
// Vectorized Threads: 24, Time: 0.0860769 seconds

//Number of Cores: 20
//hyperthreading does not occur on my device with 8 threads
//however, if i increase num_threads to 24, i get this output:
//Threads: 24, Time: 0.72319 seconds
//With Transposed B - Threads: 24, Time: 0.18118 seconds
//So the time does decrease with hyperthreading, showing an increase in performance.

void multiplyMatrix(float a[], float b[], float c[], int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            c[i*n+j] = 0.0;
            for(int k = 0; k < n; k++){
                c[i*n+j] += a[i*n+k] * b[k*n+j];
            }
        }
    }
}


void multiplyMatrix2(float a[], float b[], float c[], int n){
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            float dot = 0.0;
            for(int k = 0; k < n; k++){
                dot += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] = dot;
        }
    }
}

//multithreaded with openMP
void multiplyMatrix3(float a[], float b[], float c[], int n){
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            float dot = 0.0;
            for(int k = 0; k < n; k++){
                dot += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] = dot;
        }
    }
}

void multiplyMatrix4(float a[], float b[], float c[], int n){
    #pragma omp parallel for
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            float dot = 0.0;
            #pragma omp simd
            for(int k = 0; k < n; k++){
                dot += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] = dot;
        }
    }
}



//HW: multithreading matrix multiplication with OpenMP function
void multiplyMatrix5(float a[], float b[], float c[], int n, int threads){
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            float dot = 0.0;
            for (int k = 0; k < n; k++){
                dot += a[i*n+k] * b[k*n+j];
            }
            c[i*n+j] = dot;
        }
    }
}

//HW: transpose matrix
void transposeMatrix(float b[], float b_trans[], int n){
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            b_trans[j*n + i] = b[i*n + j];
        }
    }
}

//HW: matrix multiplication with transpose
void multiplyMatrixTransposed(float a[], float b_trans[], float c[], int n, int threads){
    #pragma omp parallel for num_threads(threads)
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            float dot = 0.0;
            for (int k = 0; k < n; k++){
                dot += a[i*n + k] * b_trans[j*n + k];
            }
            c[i*n + j] = dot;
        }
    }
}

//HW: matrix multiplication vectorized
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

//HW: benchmarks to test multithreaded matrix multiplication
void benchmarkMatrixMultiplication(float a[], float b[], float c[], int n, int threads) {
    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrix5(a, b, c, n, threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Threads: " << threads << ", Time: " << duration.count() << " seconds\n";
}

//HW: benchmarks to test multithreaded matrix multiplication with transposed B
void benchmarkWithTransposedB(float a[], float b[], float c[], int n, int threads) {
    float* b_trans = new float[n*n]; 
    transposeMatrix(b, b_trans, n);
    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrixTransposed(a, b_trans, c, n, threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "With Transposed B - Threads: " << threads << ", Time: " << duration.count() << " seconds\n";
    delete[] b_trans; 
}

void benchmarkMatrixVectorized(float a[], float b[], float c[], int n, int threads) {
    auto start = std::chrono::high_resolution_clock::now();
    multiplyMatrixVectorized(a, b, c, n, threads);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Vectorized Threads: " << threads << ", Time: " << duration.count() << " seconds\n";
}


int main() {
    int num_threads = omp_get_num_procs();
    int n = 1024;
    float* a = new float[n*n];
    float* b = new float[n*n];
    float* c = new float[n*n];
    int threadsArray[] = {1, 2, 4, 8, 24};

    //set the matrices to random values
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            a[i*n + j] = (i==j) ? 1.0f : 0.0f;
            b[i*n + j] = (i==j) ? 1.0f : 0.0f;
        }
    }

    for (int threads : threadsArray) {
        benchmarkMatrixMultiplication(a, b, c, n, threads);
    }

    // Benchmark with transposition
    for (int threads : threadsArray) {
        benchmarkWithTransposedB(a, b, c, n, threads);
    }

    for (int threads : threadsArray) {
        benchmarkMatrixVectorized(a, b, c, n, threads);
    }

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;

}