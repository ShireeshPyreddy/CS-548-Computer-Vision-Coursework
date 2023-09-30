import numpy as np
import cv2
import math as m
import gradio as gr

def create_unnormalized_hist(image):
    hist_array = np.zeros(256, dtype=np.float32)
    
    count_values = {}
    
    for each_row in image:
        for each_col in each_row:
            # if each_col in count_values:
            #     count_values[each_col] += 1
            # else:
            #     count_values[each_col] = 1
            
            hist_array[each_col] += 1

    # for key, value in count_values.items():
        # hist_array[key] = value
    
    return hist_array


def normalize_hist(hist):
    total_pixels_count = np.sum(hist)
    
    normalized_hist = hist / total_pixels_count
    
    return normalized_hist
    
def create_cdf(nhist):
    cdf_array = np.zeros(256, dtype=np.float32)
    
    cdf_array[0] = nhist[0]
    for i in range(1, len(nhist)):
        cdf_array[i] = cdf_array[i - 1] + nhist[i]
    
    return cdf_array

def constrast_limit(hist, threshold):
    extra_sum = 0
        
    for each_index, each_element in enumerate(hist):
        if each_element > threshold:
            extra = each_element - threshold
            extra_sum += extra
            hist[each_index] = threshold
    
    print(extra_sum, hist.shape, len(hist))
    
    redist = extra_sum // len(hist)
    residual = extra_sum % len(hist)
    
    for each_index in range(len(hist)):
        hist[each_index] = hist[each_index] + redist
    
    if residual:        
        residual_step  = int(max(len(hist)//residual, 1))
        
        for index in range(0, len(hist), residual_step):
            hist[index] += 1
            residual -= 1
            if residual <= 0:
                break        
    
    return hist        
    
def get_hist_equalize_transform(image, do_stretching, do_cl=False, cl_thresh=0):
    hist = create_unnormalized_hist(image)
    if do_cl:
        hist = constrast_limit(hist=hist, threshold=cl_thresh)
        
    normalized_hist = normalize_hist(hist=hist)
    cdf_array = create_cdf(nhist=normalized_hist)
    
    if do_stretching:
        starting_value = cdf_array[0]
        cdf_array = cdf_array - starting_value
        ending_value = cdf_array[-1]
        cdf_array = cdf_array / ending_value
            
    int_transform = cdf_array * 255.0
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]
    
    return int_transform

def do_histogram_equalize(image, do_stretching):
    output = np.copy(image)
    transformation_func = get_hist_equalize_transform(output, do_stretching=do_stretching)
    
    for each_row_index in range(len(image)):
        for each_col_index in range(len(image[each_row_index])):
            curr_value = image[each_row_index][each_col_index]
            new_value = transformation_func[curr_value]
            output[each_row_index][each_col_index] = new_value
        
    return output

def clamp(coords, min_val, max_val):
    return np.minimum(np.maximum(coords, min_val), max_val)

def get_block_index(coords, cnt):
    return cnt * coords[0] + coords[1]

def get_u_v_coords(br, bc):
    u = bc - m.floor(bc)
    v = br - m.floor(br)
    return u, v

def do_adaptive_histogram_equalize(image, block_cnt, cl_thresh):
    output = np.zeros(image.shape, dtype=np.float32)
    
    bw = image.shape[1]//block_cnt
    bh = image.shape[0]//block_cnt
    
    all_transforms = []
    
    for br in range(block_cnt):
        for bc in range(block_cnt):
    
            sr = br*bh
            er = sr + bh
            sc = bc*bw
            ec = sc + bw
    
            sub_image = image[sr:er, sc:ec]
            
            transformation_func = get_hist_equalize_transform(sub_image, 
                                                              do_stretching=True, 
                                                              do_cl=True, 
                                                              cl_thresh=cl_thresh)
            all_transforms.append(transformation_func)
    
    for each_row_index in range(len(image)):
        for each_col_index in range(len(image[each_row_index])):
            curr_value = image[each_row_index][each_col_index]
            br = each_row_index/bh
            bc = each_col_index/bw
            br -= 0.5
            bc -= 0.5
            
            br_floor, br_ceil = m.floor(br), m.ceil(br)
            bc_floor, bc_ceil = m.floor(bc), m.ceil(bc)
            
            upleft_index = clamp([br_floor, bc_floor], 0, block_cnt - 1)
            upright_index = clamp([br_floor, bc_ceil], 0, block_cnt - 1)
            downleft_index = clamp([br_ceil, bc_floor], 0, block_cnt - 1)
            downright_index = clamp([br_ceil, bc_ceil], 0, block_cnt - 1)
            
            upleft_index = get_block_index(upleft_index, block_cnt)
            upright_index = get_block_index(upright_index, block_cnt)
            downleft_index = get_block_index(downleft_index, block_cnt)
            downright_index = get_block_index(downright_index, block_cnt)
            
            upleft_transform = all_transforms[upleft_index]
            upright_transform = all_transforms[upright_index]
            downleft_transform = all_transforms[downleft_index]
            downright_transform = all_transforms[downright_index]
            
            new_value_upleft = upleft_transform[curr_value]
            new_value_upright = upright_transform[curr_value]
            new_value_downleft = downleft_transform[curr_value]
            new_value_downright = downright_transform[curr_value]
            
            u, v = get_u_v_coords(br, bc)
            
            new_val = ((1-v) * (1-u) * new_value_upleft
                        + (1-v) * (u) * new_value_upright
                        + (v) * (1-u) * new_value_downleft
                        + (v) * (u) * new_value_downright)

            output[each_row_index][each_col_index] = new_val
            
    
    output = cv2.convertScaleAbs(output)

    return output

def grad_intensity_callback(input_img, equal_type, block_cnt, cl_thresh):
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    if equal_type == "regular":
        output_img = do_histogram_equalize(input_img, False)
    elif equal_type == "stretching":
        output_img = do_histogram_equalize(input_img, True)
    elif equal_type == "adaptive":
        output_img = do_adaptive_histogram_equalize(input_img, int(block_cnt), int(cl_thresh))
    else:
        output_img = input_img
    return output_img

def grad_main():
    demo = gr.Interface(fn=grad_intensity_callback, 
                        inputs=["image", 
                                gr.Dropdown(choices=["regular", "stretching", "adaptive"], value="regular"), 
                                gr.Number(value=8), 
                                gr.Number(value=40)],
                        outputs=["image"])
    demo.launch() 


if __name__ == '__main__':   
    grad_main()
