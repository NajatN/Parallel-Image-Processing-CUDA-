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