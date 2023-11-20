import cv2
import numpy as np
from General_A04 import LBP_LABEL_TYPES


def create_unnormalized_hist(image, size):
    hist_array = np.zeros(size, dtype=np.float32)
    
    for each_row in image:
        for each_col in each_row:
            hist_array[each_col] += 1
    
    return hist_array

def normalize_hist(hist):
    total_pixels_count = np.sum(hist)
    
    normalized_hist = hist / total_pixels_count
    
    return normalized_hist
    
def getOneLBPLabel(subimage, label_type):   
     
    center = subimage[1, 1]
    neighbors = [subimage[0, 0], subimage[0, 1], subimage[0, 2],
                 subimage[1, 2], subimage[2, 2], subimage[2, 1],
                 subimage[2, 0], subimage[1, 0]]
    
    binary_pattern = [int(x > center) for x in neighbors]
    
    def check_uniform(pattern):
        temp = []
        for i in range(len(pattern[1:])):
            if pattern[i-1] != pattern[i]:
                temp.append(1)
            else:
                temp.append(0)
                
        if sum(temp) <= 2:
            return sum(binary_pattern)
        else:
            return 9

    if label_type == LBP_LABEL_TYPES.UNIFORM:        
        return check_uniform(binary_pattern)
    
    elif label_type == LBP_LABEL_TYPES.FULL:
        temp = ""
        for each_value in binary_pattern[::-1]:
            temp += str(each_value)
            
        return int(temp, 2)
    
    elif label_type == LBP_LABEL_TYPES.UNIFORM_ROT:
        if check_uniform(binary_pattern) == 9:
            return 58
        
        first_swap = -1
        for i in range(len(binary_pattern[:-1])):
            if binary_pattern[i] == 0 and binary_pattern[i+1] == 1:
                first_swap = i
                break
        
        if binary_pattern[-1] == 0 and binary_pattern[0] == 1:
            first_swap = len(binary_pattern)-1
        
        if sum(binary_pattern) in [0, 8] or first_swap == 0:
            return sum(binary_pattern)
        else:
            return first_swap * 7 + 1 + sum(binary_pattern)
        
def getLBPImage(image, label_type):
    
    padded_image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

    lbp_image = np.zeros_like(image)

    for each_row in range(1, image.shape[0] + 1):
        for each_col in range(1, image.shape[1] + 1):
            subimage = padded_image[each_row - 1:each_row + 2, each_col - 1:each_col + 2]
            lbp_label = getOneLBPLabel(subimage, label_type)
            lbp_image[each_row - 1, each_col - 1] = lbp_label
    
    return lbp_image

def getOneRegionLBPFeatures(subImage, label_type):
    if label_type == LBP_LABEL_TYPES.UNIFORM:
        unhist = create_unnormalized_hist(subImage, size=10)
        nhist = normalize_hist(unhist)
        
    elif label_type == LBP_LABEL_TYPES.FULL:
        unhist = create_unnormalized_hist(subImage, size=256)
        nhist = normalize_hist(unhist)
        
    elif label_type == LBP_LABEL_TYPES.UNIFORM_ROT:
        unhist = create_unnormalized_hist(subImage, size=59)
        nhist = normalize_hist(unhist)
    
    return nhist
        
def getLBPFeatures(featureImage, regionSideCnt, label_type):
    
    img_height, img_width = featureImage.shape
    
    subregion_width = img_width // regionSideCnt
    subregion_height = img_height // regionSideCnt
    
    all_hists = []

    for each_row in range(regionSideCnt):
        for each_col in range(regionSideCnt):
            temp_row = each_row * subregion_height
            temp_col = each_col * subregion_width

            subimage = featureImage[temp_row:temp_row + subregion_height, temp_col:temp_col + subregion_width]

            hist = getOneRegionLBPFeatures(subimage, label_type)

            all_hists.append(hist)
    
    all_hists = np.array(all_hists)
    all_hists = np.reshape(all_hists, (all_hists.shape[0] * all_hists.shape[1],))

    return all_hists
