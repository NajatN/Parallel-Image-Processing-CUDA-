# Parallel Image Processing with CUDA
Welcome to the Parallel Image Processing project utilizing CUDA for fast, efficient image processing tasks. This project harnesses the power of parallel computing to manipulate images in various ways. CUDA, which stands for Compute Unified Device Architecture, is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing, which is highly beneficial for image processing tasks due to their inherent parallel nature. So, if you're interested in CUDA, image processing, and harnessing the power of parallel computing, you've come to the right place!

## Getting Started :rocket:
This project is designed to be run on [Google Colab](https://colab.research.google.com/), a cloud-based Python environment that provides free access to GPUs. Here's a step-by-step guide to getting the project up and running.

### Step 1: Preparation :file_folder:

First, you'll need to upload your project files to Google Colab. You can do this in one of two ways:

- **Option 1:** Use the file explorer in the left-hand panel of your Colab notebook to upload your files.
- **Option 2:** Use the following Python commands in a new Colab cell:

```python
from google.colab import files
uploaded = files.upload()
```
This will create a "Choose Files" button, which you can use to upload your files. You'll need to upload all the .cu, .h, and .txt files from your local project directory.

### Step 2: Compilation and Execution :hammer_and_wrench:

Google Colab comes pre-installed with the CUDA Toolkit, which includes the nvcc compiler needed to compile CUDA code.

To compile the job_parser.cu file, create a new cell in your Colab notebook and run:

```
!nvcc -o job_parser job_parser.cu -lpng
```
This will compile your CUDA code and create an executable named job_parser.

Next, to run the compiled file, create another cell and run:

```
!./job_parser
```
This will execute your CUDA program and perform the image processing tasks.

### Step 3: Retrieving the Output :outbox_tray:
If your CUDA code generates output files (like processed images), you can download them directly from Colab.

- **Option 1:** Use the file explorer in the left-hand panel of your Colab notebook. You'll see your output files listed there. Right-click on a file and select "Download" to save it to your local machine.
- **Option 2:** Use the following Python commands in a new Colab cell:
```python
from google.colab import files
files.download('your_filename')
```
Replace 'your_filename' with the name of your output file.

## Wrapping Up :gift:
That's it! You're now ready to perform parallel image processing with CUDA on Google Colab. Enjoy your journey into the world of parallel computing and image processing! :rocket:


## Code Explanation :books:

This is a CUDA program that performs various image processing operations on PNG images. These operations include brightness adjustment, grayscale conversion, channel correction, blurring, sharpening, and edge detection. Below, we'll break down the code into its main parts.

### Header Inclusions :file_folder:

The beginning of the program includes necessary libraries for the operations performed in the code:

- stdio.h and stdlib.h for general functionality (e.g., file I/O, dynamic memory management).
- string.h for string manipulation functions.
- cuda_runtime.h and cuda_runtime_api.h for CUDA runtime.
- device_launch_parameters.h for accessing thread and block indices, block dimensions, etc.
- png.h for PNG file operations.
- sys/time.h for time measurement functionality.

### Data Structures :wrench:

Two data structures are defined:

- Job: Each Job consists of an input filename, algorithm name, and output filename. These jobs are read from a file and executed in sequence.
- Image: Each Image has a data array (the pixel data), dimensions (width and height), and the number of channels (i.e., color depth).

### Time Measurement Function :alarm_clock:

timeInMilliseconds: This function returns the current time in milliseconds. It's used to measure the execution time of each job.

### Job Handling Functions :clipboard:

read_jobs: This function reads jobs from a file. Each job is a line with an input filename, algorithm name, and output filename.

### Image Handling Functions :sunrise_over_mountains:

load_image and save_image: These functions handle reading and writing PNG images, respectively. They use the libpng library to do so.

### CUDA Kernels :dart:

There are several CUDA kernels in your code which apply various image processing operations:

- brightness_kernel: Adjusts the brightness of an image by a specified value.
- grayscale_kernel: Converts an image to grayscale.
- channel_correction_kernel: Applies channel correction factors to an image.
- apply_convolution_kernel: Applies a convolutional filter to an image. This kernel is used in blurring, sharpening, and edge detection operations.
- blurring_kernel, sharpening_kernel, edge_detection_kernel: These are not actually CUDA kernels but functions that create a filter, allocate memory on the GPU for it, copy it there, and then call apply_convolution_kernel with the appropriate parameters.

### Execution Function :rocket:

execute_jobs: This function executes the jobs read from the file. It loads the image, applies the algorithm specified in the job on the GPU, and saves the processed image. For each job, it measures and prints the execution time.
