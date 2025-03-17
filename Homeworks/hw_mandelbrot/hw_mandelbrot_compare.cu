//Author: Ritwika Das, rd935
//Cited: used ChatGPT for help

//Compile: nvcc hw_mandelbrot_compare.cu -o mandelbrot_compare -std=c++11
//Run: ./mandelbrot_compare
//Output: CPU Execution Time: 0.672958 seconds
//        GPU Execution Time: 0.0212972 seconds

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // Image write library

// CPU Mandelbrot function
uint32_t mandelbrot_cpu(float x0, float y0, uint32_t maxcount) {
    float x = 0.0f;
    float y = 0.0f;
    uint32_t count = 0;

    while (count < maxcount && (x * x + y * y) < 4.0f) {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        count++;
    }
    return count;
}

// CPU version of Mandelbrot set
void mandelbrot_cpu(uint32_t counts[], uint32_t maxcount, uint32_t w, uint32_t h, float xmin, float xmax, float ymin, float ymax) {
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float real = xmin + (xmax - xmin) * x / (w - 1);
            float imag = ymin + (ymax - ymin) * y / (h - 1);
            counts[y * w + x] = mandelbrot_cpu(real, imag, maxcount);
        }
    }
}

// GPU Mandelbrot device function
__device__ uint32_t mandelbrot_gpu(float x0, float y0, uint32_t maxcount) {
    float x = 0.0f;
    float y = 0.0f;
    uint32_t count = 0;

    while (count < maxcount && (x * x + y * y) < 4.0f) {
        float xtemp = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = xtemp;
        count++;
    }
    return count;
}

// GPU kernel for Mandelbrot set
__global__ void mandelbrot_kernel(uint32_t *counts, uint32_t maxcount, float xmin, float xmax, float ymin, float ymax, uint32_t w, uint32_t h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        float real = xmin + (xmax - xmin) * x / (w - 1);
        float imag = ymin + (ymax - ymin) * y / (h - 1);
        counts[y * w + x] = mandelbrot_gpu(real, imag, maxcount);
    }
}

// GPU version wrapper function
void mandelbrot_gpu(uint32_t counts[], uint32_t maxcount, uint32_t w, uint32_t h, float xmin, float xmax, float ymin, float ymax) {
    uint32_t *d_counts;
    size_t size = w * h * sizeof(uint32_t);
    
    cudaMalloc(&d_counts, size);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((w + blockSize.x - 1) / blockSize.x, (h + blockSize.y - 1) / blockSize.y);
    
    mandelbrot_kernel<<<gridSize, blockSize>>>(d_counts, maxcount, xmin, xmax, ymin, ymax, w, h);
    cudaMemcpy(counts, d_counts, size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_counts);
}

// Timing comparison function
void compare_cpu_gpu(uint32_t *counts_cpu, uint32_t *counts_gpu, uint32_t width, uint32_t height, uint32_t maxcount, float xmin, float xmax, float ymin, float ymax) {
    // CPU Timing
    auto start_cpu = std::chrono::high_resolution_clock::now();
    mandelbrot_cpu(counts_cpu, maxcount, width, height, xmin, xmax, ymin, ymax);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    std::cout << "CPU Execution Time: " << cpu_time.count() << " seconds\n";

    // GPU Timing
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    mandelbrot_gpu(counts_gpu, maxcount, width, height, xmin, xmax, ymin, ymax);
    cudaEventRecord(stop_gpu);

    cudaEventSynchronize(stop_gpu);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start_gpu, stop_gpu);
    gpu_time /= 1000.0;  // Convert ms to seconds

    std::cout << "GPU Execution Time: " << gpu_time << " seconds\n";

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);
}

// Main function
int main() {
    const uint32_t maxcount = 256;
    const uint32_t width = 1280;
    const uint32_t height = 1024;
    uint32_t *counts_cpu = (uint32_t *)malloc(width * height * sizeof(uint32_t));
    uint32_t *counts_gpu = (uint32_t *)malloc(width * height * sizeof(uint32_t));

    // Compare CPU and GPU
    compare_cpu_gpu(counts_cpu, counts_gpu, width, height, maxcount, -2.0f, 1.0f, -1.5f, 1.5f);

    // Free allocated memory
    free(counts_cpu);
    free(counts_gpu);
    return 0;
}
