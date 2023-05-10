#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <png.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

typedef struct {
    char input_filename[256];
    char algorithm_name[256];
    char output_filename[256];
} Job;

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
} Image;

void save_image(const char *filename, Image *image);

void read_jobs(const char *filename, Job *jobs, int *num_jobs) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening input file");
        exit(EXIT_FAILURE);
    }
    char line[1024];
    *num_jobs = 0;
    while (fgets(line, sizeof(line), file)) {
        sscanf(line, "%s %s %s", jobs[*num_jobs].input_filename, jobs[*num_jobs].algorithm_name, jobs[*num_jobs].output_filename);
        (*num_jobs)++;
    }
    fclose(file);
}

void load_image(const char *filename, Image *image) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error opening image: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        fprintf(stderr, "Error creating PNG read struct\n");
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fclose(fp);
        png_destroy_read_struct(&png, NULL, NULL);
        fprintf(stderr, "Error creating PNG info struct\n");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        png_destroy_read_struct(&png, &info, NULL);
        fprintf(stderr, "Error during PNG init_io\n");
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);
    png_read_info(png, info);

    image->width = png_get_image_width(png, info);
    image->height = png_get_image_height(png, info);
    image->channels = png_get_channels(png, info);

    if (image->channels == 4) {
        png_set_strip_alpha(png);
        image->channels = 3;
    }

    png_read_update_info(png, info);

    image->data = (unsigned char *)malloc(image->width * image->height * image->channels);
    if (!image->data) {
        fclose(fp);
        png_destroy_read_struct(&png, &info, NULL);
        fprintf(stderr, "Error allocating memory for the image\n");
        exit(EXIT_FAILURE);
    }

    png_bytep *row_pointers = (png_bytep *)malloc(image->height * sizeof(png_bytep));
    for (int y = 0; y < image->height; y++) {
        row_pointers[y] = (png_byte *)(image->data + y * image->width * image->channels);
    }

    png_read_image(png, row_pointers);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    free(row_pointers);

    printf("Loaded image: %s (%d x %d x %d)\n", filename, image->width, image->height, image->channels);
}

__global__ void brightness_kernel(Image img_in, Image img_out, int value) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_in.width && y < img_in.height) {
        int idx = (y * img_in.width + x) * img_in.channels;
        for (int c = 0; c < img_in.channels; c++) {
            img_out.data[idx + c] = max(0, min(255, img_in.data[idx + c] + value));
        }
    }
}

__global__ void grayscale_kernel(Image img_in, Image img_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_in.width && y < img_in.height) {
        int idx = (y * img_in.width + x) * img_in.channels;
        float gray = 0.299f * img_in.data[idx] + 0.587f * img_in.data[idx + 1] + 0.114f * img_in.data[idx + 2];
        unsigned char gray_int = max(0, min(255, (int)gray));
        img_out.data[idx] = gray_int;
        img_out.data[idx + 1] = gray_int;
        img_out.data[idx + 2] = gray_int;
    }
}

__global__ void channel_correction_kernel(Image img, float red, float green, float blue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img.width && y < img.height) {
        int idx = (y * img.width + x) * img.channels;
        img.data[idx] = max(0, min(255, (int)(img.data[idx] * red)));
        img.data[idx + 1] = max(0, min(255, (int)(img.data[idx + 1] * green)));
        img.data[idx + 2] = max(0, min(255, (int)(img.data[idx + 2] * blue)));
    }
}

__global__ void apply_convolution_kernel(Image img_in, Image img_out, const float * __restrict__ filter, int filter_width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < img_in.width && y < img_in.height) {
        int idx = (y * img_in.width + x) * img_in.channels;
        float sum[3] = {0.0f, 0.0f, 0.0f};

        for (int ky = -filter_width / 2; ky <= filter_width / 2; ky++) {
            for (int kx = -filter_width / 2; kx <= filter_width / 2; kx++) {
                int ix = min(max(x + kx, 0), img_in.width - 1);
                int iy = min(max(y + ky, 0), img_in.height - 1);
                int k_idx = (ky + filter_width / 2) * filter_width + (kx + filter_width / 2);
                int img_idx = (iy * img_in.width + ix) * img_in.channels;

                for (int c = 0; c < img_in.channels; c++) {
                    sum[c] += img_in.data[img_idx + c] * filter[k_idx];
                }
            }
        }

        for (int c = 0; c < img_in.channels; c++) {
            img_out.data[idx + c] = max(0, min(255, (int)sum[c]));
        }
    }
}

void blurring_kernel(Image img_in, Image img_out, int kernel_size, dim3 blockDim, dim3 gridDim) {
    float *filter = new float[kernel_size * kernel_size];
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        filter[i] = 1.0f / (kernel_size * kernel_size);
    }
    
    float *filter_gpu;
    cudaMalloc((void **)&filter_gpu, kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(filter_gpu, filter, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    apply_convolution_kernel<<<gridDim, blockDim>>>(img_in, img_out, filter_gpu, kernel_size);
    
    cudaFree(filter_gpu);
    delete[] filter;
}

void sharpening_kernel(Image img_in, Image img_out, int kernel_size, dim3 blockDim, dim3 gridDim) {
    float *filter = new float[kernel_size * kernel_size];
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        filter[i] = (i == kernel_size * kernel_size / 2) ? kernel_size * kernel_size - 1 : -1;
    }

    float *filter_gpu;
    cudaMalloc((void **)&filter_gpu, kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(filter_gpu, filter, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    apply_convolution_kernel<<<gridDim, blockDim>>>(img_in, img_out, filter_gpu, kernel_size);
    
    cudaFree(filter_gpu);
    delete[] filter;
}

void edge_detection_kernel(Image img_in, Image img_out, int kernel_size, dim3 blockDim, dim3 gridDim) {
    float *filter = new float[kernel_size * kernel_size];
    for (int i = 0; i < kernel_size * kernel_size; i++) {
        filter[i] = (i == kernel_size * kernel_size / 2) ? -(kernel_size * kernel_size - 1) : 1;
    }
    
    float *filter_gpu;
    cudaMalloc((void **)&filter_gpu, kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(filter_gpu, filter, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    apply_convolution_kernel<<<gridDim, blockDim>>>(img_in, img_out, filter_gpu, kernel_size);
    
    cudaFree(filter_gpu);
    delete[] filter;
}

void debug_image(const char *filename, Image *image) {
    unsigned char *cpu_data = new unsigned char[image->width * image->height * image->channels];
    cudaMemcpy(cpu_data, image->data, image->width * image->height * image->channels, cudaMemcpyDeviceToHost);

    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error opening image: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) {
        fclose(fp);
        fprintf(stderr, "Error creating PNG write struct\n");
        exit(EXIT_FAILURE);
    }

    png_infop info = png_create_info_struct(png);
    if (!info) {
        fclose(fp);
        png_destroy_write_struct(&png, NULL);
        fprintf(stderr, "Error creating PNG info struct\n");
        exit(EXIT_FAILURE);
    }

    if (setjmp(png_jmpbuf(png))) {
        fclose(fp);
        png_destroy_write_struct(&png, &info);
        fprintf(stderr, "Error during PNG init_io\n");
        exit(EXIT_FAILURE);
    }

    png_init_io(png, fp);

    png_set_IHDR(png,
                 info,
                 image->width,
                 image->height,
                 8,
                 image->channels == 3 ? PNG_COLOR_TYPE_RGB : PNG_COLOR_TYPE_RGBA,
                 PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png, info);

    png_bytep *row_pointers = (png_bytep *)malloc(image->height * sizeof(png_bytep));
    for (int y = 0; y < image->height; y++) {
        row_pointers[y] = (png_byte *)(cpu_data + y * image->width * image->channels);
    }

    png_write_image(png, row_pointers);
    png_write_end(png, NULL);

    fclose(fp);
    png_destroy_write_struct(&png, &info);
    free(row_pointers);

    printf("Saved image: %s (%d x %d x %d)\n", filename, image->width, image->height, image->channels);

    delete[] cpu_data;
}

void execute_jobs(Job *jobs, int num_jobs) {
    for (int i = 0; i < num_jobs; i++) {
        Image img_in, img_out, img_in_cpu;
        load_image(jobs[i].input_filename, &img_in_cpu);

        // Allocate GPU memory for the input image and copy the data
        cudaMalloc((void **)&img_in.data, img_in_cpu.width * img_in_cpu.height * img_in_cpu.channels);
        cudaMemcpy(img_in.data, img_in_cpu.data, img_in_cpu.width * img_in_cpu.height * img_in_cpu.channels, cudaMemcpyHostToDevice);
        
        img_in.width = img_in_cpu.width;
        img_in.height = img_in_cpu.height;
        img_in.channels = img_in_cpu.channels;

        // Allocate GPU memory for the output image
        cudaMalloc((void **)&img_out.data, img_in.width * img_in.height * img_in.channels);
        cudaMemcpy(img_out.data, img_in.data, img_in.width * img_in.height * img_in.channels, cudaMemcpyHostToDevice);
        img_out.width = img_in.width;
        img_out.height = img_in.height;
        img_out.channels = img_in.channels;

        dim3 blockDim(32, 32);
        dim3 gridDim((img_in.width + blockDim.x - 1) / blockDim.x, (img_in.height + blockDim.y - 1) / blockDim.y);

        if (strcmp(jobs[i].algorithm_name, "brightness") == 0) {
            int value = 50; // Set the desired brightness value
            brightness_kernel<<<gridDim, blockDim>>>(img_in, img_out, value);
        } else if (strcmp(jobs[i].algorithm_name, "grayscale") == 0) {
            grayscale_kernel<<<gridDim, blockDim>>>(img_in, img_out);
        } else if (strcmp(jobs[i].algorithm_name, "channel_correction") == 0) {
            float red = 1.0f, green = 1.0f, blue = 1.0f; // Set the desired channel correction factors
            channel_correction_kernel<<<gridDim, blockDim>>>(img_in, red, green, blue);
        } else if (strcmp(jobs[i].algorithm_name, "blurring") == 0) {
            int kernel_size = 5; // Set the desired kernel size for the blurring algorithm
            blurring_kernel(img_in, img_out, kernel_size, blockDim, gridDim);
        } else if (strcmp(jobs[i].algorithm_name, "sharpening") == 0) {
            int kernel_size = 3; // Set the desired kernel size for the sharpening algorithm
            sharpening_kernel(img_in, img_out, kernel_size, blockDim, gridDim);
        } else if (strcmp(jobs[i].algorithm_name, "edge_detection") == 0) {
            int kernel_size = 3; // Set the desired kernel size for the edge detection algorithm
            edge_detection_kernel(img_in, img_out, kernel_size, blockDim, gridDim);
        }
        cudaDeviceSynchronize();

        // Save the processed image and deallocate GPU memory
        debug_image(jobs[i].output_filename, &img_out);
        cudaFree(img_in.data);
        cudaFree(img_out.data);
        free(img_in_cpu.data);
    }
}

void save_image(const char *filename, Image *image) {
    // Copy image data from GPU to CPU before saving
    unsigned char *cpu_data = new unsigned char[image->width * image->height * 3];
    cudaMemcpy(cpu_data, image->data, image->width * image->height * 3, cudaMemcpyDeviceToHost);

    printf("Saving image: %s (%d x %d x %d)\n", filename, image->width, image->height, 3);

    // Save the image using stb_image_write
    if (!stbi_write_png(filename, image->width, image->height, 3, cpu_data, image->width * 3)) {
        fprintf(stderr, "Error saving image: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    delete[] cpu_data;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    const char *input_filename = argv[1];
    Job jobs[1024];
    int num_jobs = 0;

    read_jobs(input_filename, jobs, &num_jobs);
    execute_jobs(jobs, num_jobs);

    return 0;
}