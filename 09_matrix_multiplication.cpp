#include <omp.h>

//g++ -O2 -g -fopenmp 09_matrix_multiplication.cpp

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

//HW: make a better matrix multiply where second matrix is transposing

int main() {
    int num_threads = omp_get_num_procs();
    int n = 1024;
    float* a = new float[num_threads];
    float* b = new float[num_threads];
    float* c = new float[num_threads];

    //set the matrices to random values
    for (int i = 0; i < n*n; i++){
        a[i] = 1;
        b[i] = 1;
    }

    multiplyMatrix(a, b, c, n);
    multiplyMatrix2(a, b, c, n);

    multiplyMatrix3(a, b, c, n);

    delete[] a;
    delete[] b;
    delete[] c;
    return 0;

}