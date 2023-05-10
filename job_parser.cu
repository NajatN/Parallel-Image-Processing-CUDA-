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