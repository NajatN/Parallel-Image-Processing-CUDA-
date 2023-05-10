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