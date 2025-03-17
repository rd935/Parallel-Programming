//Author: Ritwika Das, rd935
//Cited: used ChatGPT for help

//Compile: g++ hw_mandelbrot_cpu.cpp -o mandelbrot_cpu -lm
//Run: ./mandelbrot_cpu
//Output: mandebrot_cpu.png

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"  // Include the image write library

// Function to compute the number of iterations for each point in the Mandelbrot set
uint32_t mandelbrot(float x0, float y0, uint32_t maxcount) {
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

// Function to generate the Mandelbrot set counts
void mandelbrot(uint32_t counts[], uint32_t maxcount, uint32_t w, uint32_t h, float xmin, float xmax, float ymin, float ymax) {
    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            float real = xmin + (xmax - xmin) * x / (w - 1);
            float imag = ymin + (ymax - ymin) * y / (h - 1);
            counts[y * w + x] = mandelbrot(real, imag, maxcount);
        }
    }
}

// Function to convert counts to color and generate the image
void generate_image(uint32_t *counts, uint32_t w, uint32_t h, const char *filename) {
    uint8_t *image = (uint8_t *)malloc(w * h * 3);  // Allocate space for RGB image

    for (uint32_t y = 0; y < h; ++y) {
        for (uint32_t x = 0; x < w; ++x) {
            uint32_t count = counts[y * w + x];
            
            // If count reached max, set it to black (indicating it is inside the Mandelbrot set)
            if (count == 256) {
                image[(y * w + x) * 3 + 0] = 0;    // Red
                image[(y * w + x) * 3 + 1] = 0;    // Green
                image[(y * w + x) * 3 + 2] = 0;    // Blue
            } else {
                // Map count to a color gradient
                uint8_t r = (uint8_t)(255 * (count % 32) / 31);  // Red gradient
                uint8_t g = (uint8_t)(255 * (count % 64) / 63);  // Green gradient
                uint8_t b = (uint8_t)(255 * (count % 128) / 127); // Blue gradient
                
                image[(y * w + x) * 3 + 0] = r;
                image[(y * w + x) * 3 + 1] = g;
                image[(y * w + x) * 3 + 2] = b;
            }
        }
    }

    stbi_write_png(filename, w, h, 3, image, w * 3);  // Save as PNG with proper stride
    free(image);
}

// Main function
int main() {
    const uint32_t maxcount = 256;  // Maximum iterations
    const uint32_t width = 1280;    // Image width
    const uint32_t height = 1024;   // Image height
    uint32_t *counts = (uint32_t *)malloc(width * height * sizeof(uint32_t));

    // Generate Mandelbrot set counts
    mandelbrot(counts, maxcount, width, height, -2.0f, 1.0f, -1.5f, 1.5f);

    // Generate and save the image
    generate_image(counts, width, height, "mandelbrot_cpu.png");

    // Free allocated memory
    free(counts);
    return 0;
}
