# Parallel Image Processing (CUDA)

## Steps for Running the Code on Google Colab
_____________
Google Colab provides a free-to-use environment equipped with GPUs that support CUDA. Here's a step-by-step guide on how to run this project on Google Colab:

### Step 1: Upload Your Files

You can upload files to Google Colab using the file explorer in the left panel of your notebook. Alternatively, you can run the following Python commands in a cell:

```python
from google.colab import files
uploaded = files.upload()
```
This will create a "Choose Files" button, which you can use to upload your files.

### Step 2: Run the CUDA Code
To compile your CUDA code, you can use the nvcc command. For example, to compile the job_parser.cu file, you would run:

```
!nvcc -o job_parser job_parser.cu -lpng
```

To run the compiled file, you would use:

```
!./job_parser
```

### Step 4: Download the Processed Images
If your CUDA code generates output files, you can download them using the file explorer in the left panel. Alternatively, you can run the following Python commands in a cell:

```python
from google.colab import files
files.download('your_filename')
```
Replace 'your_filename' with the name of your output file.

That's it! You're now ready to run this CUDA image processing code on Google Colab.