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
    image = image.astype("float64")
    kernel = kernel.astype("float64")
    
    kernel = cv2.flip(kernel, -1)
    
    kernel_height = kernel.shape[0] // 2
    kernel_width = kernel.shape[1] // 2
    
    padded_image = cv2.copyMakeBorder(image, 
                                      kernel_height, 
                                      kernel_height, 
                                      kernel_width, 
                                      kernel_width, 
                                      cv2.BORDER_CONSTANT, 
                                      value=0)
    
    output_image = np.zeros_like(image, 
                                 dtype=np.float64)
    
    for each_row_index in range(len(image)):
        for each_col_index in range(len(image[each_row_index])):
            sub_image = padded_image[each_row_index:each_row_index + kernel.shape[0], 
                                     each_col_index:each_col_index + kernel.shape[1]]
            
            filter_vals = sub_image * kernel
            
            value = np.sum(filter_vals)
            
            output_image[each_row_index, each_col_index] = value
    
    if convert_uint8 is True:
        output_image = cv2.convertScaleAbs(output_image,
                                           alpha=alpha,
                                           beta=beta)
    
    return output_image


def check_zero_cross(v1, v2, thresh):
    if (v1 * v2) >= 0:
        return False
    elif m.fabs(v1 - v2) < thresh:
        return False
    else:
        return True

def get_marr_hildreth_edges(image, scale, thresh):
    gauss_1D = cv2.getGaussianKernel(scale, -1)
    
    gauss_1D_T = np.transpose(gauss_1D)
    
    blur_image_1 = apply_filter(image, 
                                gauss_1D, 
                                convert_uint8=False)
    
    blur_image_2 = apply_filter(blur_image_1, 
                                gauss_1D_T, 
                                convert_uint8=False)
    
    lap = np.array([[0,1,0],[1,-4,1],[0,1,0]], 
                   dtype="float64")
    
    lap_image = apply_filter(blur_image_2, 
                             lap, 
                             convert_uint8=False)
    
    destination_image = np.zeros_like(lap_image, 
                                      dtype=np.uint8)
    
    for each_row_index in range(len(lap_image[:-1])):
        for each_col_index in range(len(lap_image[each_row_index][:-1])):
            upper_left = lap_image[each_row_index][each_col_index]
            upper_right = lap_image[each_row_index][each_col_index + 1]
            lower_left = lap_image[each_row_index + 1][each_col_index]
            lower_right = lap_image[each_row_index + 1][each_col_index + 1]
            
            if check_zero_cross(upper_left, upper_right, thresh) or \
               check_zero_cross(upper_left, lower_left, thresh) or \
               check_zero_cross(upper_left, lower_right, thresh) or \
               check_zero_cross(lower_left, upper_right, thresh):
                destination_image[each_row_index, each_col_index] = 255
    
    return destination_image


def filtering_callback(input_img, filter_file, alpha_val, beta_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    kernel = read_kernel_file(filter_file.name)
    output_img = apply_filter(input_img, kernel, alpha_val, beta_val) 
    return output_img


def edge_callback(input_img, scale_val, thresh_val):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    output_img = get_marr_hildreth_edges(input_img, scale_val, thresh_val) 
    return output_img


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                image_data = gr.Image(label="Input Image")

            with gr.Tab("Filtering"): 
                with gr.Row(): 
                    with gr.Column(): 
                        filter_filename = gr.File(label="Filter File")
                        alpha_number = gr.Number(label="Alpha", value=0.125)
                        beta_number = gr.Number(label="Beta", value=127)
                        filter_button = gr.Button("Perform Filtering")
                    image_output = gr.Image(label="Filtered Image")

            with gr.Tab("Marr-Hildreth"): 
                with gr.Row(): 
                    with gr.Column(): 
                        scale_number = gr.Number(label="Scale", value=7, precision=0)
                        thresh_number = gr.Number(label="Threshold", value=3, precision=0)
                        edge_button = gr.Button("Get Marr-Hildreth Edges")
                    mh_output = gr.Image(label="Edge Image")
                    
        filter_button.click(filtering_callback, 
                            inputs=[image_data,filter_filename, alpha_number, beta_number], 
                            outputs=image_output)

        edge_button.click(edge_callback, 
                          inputs=[image_data, scale_number, thresh_number], 
                          outputs=mh_output)
        
    demo.launch() 
                    
    
# Later, at the bottom
if __name__ == "__main__": 
    main()
    
