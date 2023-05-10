# Parallel Image Processing with CUDA
Welcome to the Parallel Image Processing project utilizing CUDA for fast, efficient image processing tasks. This project harnesses the power of parallel computing to manipulate images in various ways. If you're interested in CUDA and image processing, you've come to the right place!

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
```
from google.colab import files
files.download('your_filename')
```
Replace 'your_filename' with the name of your output file.

## Wrapping Up :gift:
That's it! You're now ready to perform parallel image processing with CUDA on Google Colab. Enjoy your journey into the world of parallel computing and image processing! :rocket:
