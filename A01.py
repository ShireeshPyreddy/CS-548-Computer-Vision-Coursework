import numpy as np
import cv2
import matplotlib.pyplot as plt 

def read_image():
    sample = cv2.imread("assign01\images\image07.png")    
    sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    
    return sample

def create_unnormalized_hist(image):
    hist_array = np.zeros(256, dtype=np.float32)
    
    # print(hist_array)
    # print(image)
    
    count_values = {}
    
    for each_row in image:
        for each_col in each_row:
            if each_col in count_values:
                count_values[each_col] += 1
            else:
                count_values[each_col] = 1

    for key, value in count_values.items():
        hist_array[key] = value
    
    # print(hist_array)
    
    # plt.bar(range(256), hist_array, width=1.0, align='edge')
    # plt.xlabel('Gray Value (r)')
    # plt.ylabel('Probability')
    # plt.title('Normalized Histogram (Probability of Gray Values)')
    # plt.show()
    
    return hist_array


def normalize_hist(hist):
    total_pixels_count = np.sum(hist)
    
    normalized_hist = hist / total_pixels_count
    
    # print(normalized_hist)

    # plt.bar(range(256), normalized_hist, width=1.0, align='edge')
    # plt.xlabel('Gray Value (r)')
    # plt.ylabel('Probability')
    # plt.title('Normalized Histogram (Probability of Gray Values)')
    # plt.show()
    
    return normalized_hist
    
def create_cdf(nhist):
    cdf_array = np.zeros(256, dtype=np.float32)
    
    cdf_array[0] = nhist[0]
    for i in range(1, len(nhist)):
        cdf_array[i] = cdf_array[i - 1] + nhist[i]
    
    # print(cdf_array[-1])
    
    # plt.bar(range(256), cdf_array, width=1.0, align='edge')
    # plt.xlabel('Gray Value (r)')
    # plt.ylabel('Probability')
    # plt.title('Normalized Histogram (Probability of Gray Values)')
    # plt.show()
    
    return cdf_array

def constrast_limit(hist, threshold):
    extra_sum = 0
    
    # print("BEFORE:::", hist)
    
    for each_index, each_element in enumerate(hist):
        if each_element > threshold:
            extra = each_element - threshold
            extra_sum += extra
            hist[each_index] = threshold

    # print("AFTER:::", hist)
    
    print(extra_sum, hist.shape, len(hist))
    
    redist = extra_sum // len(hist)
    residual = extra_sum % len(hist)

    # print(residual)
    
    for each_index in range(len(hist)):
        hist[each_index] = hist[each_index] + redist
    
    if residual != 0:        
        residual_step  = int(max(len(hist)//residual, 1))
        
        for index in range(0, len(hist), residual_step):
            hist[index] += 1
            residual -= 1
            if residual <= 0:
                break        

    print("+++", hist)
    
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
        
        stretched_image = cdf_array[image]
        
        print(stretched_image)
    
    int_transform = cdf_array * 255.0
    int_transform = cv2.convertScaleAbs(int_transform)[:,0]

    print(int_transform)
    
    return int_transform

def do_histogram_equalize(image, do_stretching):
    output = np.copy(image)
    tranformation_func = get_hist_equalize_transform(output, do_stretching=do_stretching)
    
    for each_row_index in range(len(image)):
        for each_col_index in range(len(image[each_row_index])):
            value = image[each_row_index][each_col_index]
            new_value = tranformation_func[value]
            output[each_row_index][each_col_index] = new_value
    
    print(output)
    
    return output

def clamp(coords, min_val, max_val):
    pass

def get_block_index(coords, cnt):
    pass

def get_u_v_coords(br, bc):
    pass

def do_adaptive_histogram_equalize(image, block_cnt, cl_thresh):
    output = np.zeros(image.shape, dtype=np.float32)
    
    bw = image.shape[1]//block_cnt
    bh = image.shape[0]//block_cnt
    
    print(bw)
    
    print(bh)
    
    sr = br*bh
    er = sr + bh
    sc = bc*bw
    ec = sc + bw
    
    subimage = image[sr:er, sc:ec]
    
    all_transforms = []
    

if __name__ == '__main__':    
    sample_image = read_image()
    hist = create_unnormalized_hist(sample_image)
    # normalized_hist = normalize_hist(hist=hist)
    # create_cdf(normalized_hist)
    # constrast_limit(hist=hist, threshold=40)
    # get_hist_equalize_transform(sample_image, do_stretching=False, do_cl=True, cl_thresh=40)
    # do_histogram_equalize(sample_image, do_stretching=True)
    do_adaptive_histogram_equalize(sample_image, block_cnt=3, cl_thresh=40)
