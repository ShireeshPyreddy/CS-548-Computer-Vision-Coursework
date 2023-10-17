import cv2
import numpy as np
import gradio as gr
import math as m

def read_kernel_file(filepath):
    with open(filepath, 'r') as kernel_values:
        kernel_details = kernel_values.read()
    
    row_count = int(kernel_details.split()[0])
    col_count = int(kernel_details.split()[1])
    
    kernel = np.zeros((row_count, col_count))
    
    count = 2
    
    for each_row in range(row_count):
        for each_col in range(col_count):
            kernel[each_row][each_col] = float(kernel_details.split()[count])
            count += 1
    
    return kernel


def apply_filter(image, kernel, alpha=1.0, beta=0.0, convert_uint8=True):
    pass


def check_zero_cross(v1, v2, thresh):
    pass


def get_marr_hildreth_edges(image, scale, thresh):
    pass

