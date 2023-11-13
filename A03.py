import numpy as np
from skimage.segmentation import slic
import joblib
import cv2


pca_fit = joblib.load('feature_reduction.pkl')
model = joblib.load('image_classifier.pkl')

def resize_to_closest_square_size(image, desired_size):
    """
    resize to the closest size to our desired one
    """
    old_width, old_height = image.shape
    ratio = float(desired_size) / max(old_width, old_height)
    new_width, new_height = int(old_width * ratio), int(old_height * ratio)
    image = cv2.resize(image, (new_width, new_height))
    return (image, new_width, new_height)

def resize_with_padding(image, desired_size, border_type = cv2.BORDER_REPLICATE):
    image, new_width, new_height = resize_to_closest_square_size(image, desired_size)

    delta_width = desired_size - new_width
    delta_height = desired_size - new_height

    top = delta_height//2
    bottom = delta_height - top
    left = delta_width//2
    right = delta_width - left
 
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(image, top, bottom, left, right, border_type, value=color)
    return new_image

def preprocess(image):
    image_with_increased_contrast = increase_contrast(image)

    gray_image = cv2.cvtColor(image_with_increased_contrast, cv2.COLOR_BGR2GRAY)
  
    return gray_image

def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    l2 = clahe.apply(l)

    lab = cv2.merge((l2,a,b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) 
    
    return img2

def crop_image(image, xmin, xmax, ymin, ymax):
    # check for incorrect annotations of the bounding box
    if(ymax - ymin > 0 and xmax - xmin > 0):
        cropped_image = image[ymin:ymax, xmin:xmax]
        return cropped_image
    else: 
        return None


def find_WBC(image):
    """
    To increase accuracy, finetuned VGG16 model using tensorflow and got 0.38 IOU with over 90% accuracy.
    The training code can be found in Train_WBC.py file.
    
    But below method outperformed the above deeplearning model. Hence continued with the below method.
    """
    pixels = image.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4  
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    target_color = np.array([255, 0, 0], dtype=np.float32)
    closest_cluster_index = np.argmin(np.linalg.norm(centers - target_color, axis=1))
    closest_cluster_mask = (labels.flatten() == closest_cluster_index).astype(np.uint8) * 255
    mask_image = closest_cluster_mask.reshape(image.shape[:2])
    num_labels, labels_im = cv2.connectedComponents(mask_image)

    bounding_boxes = []
    for i in range(1, num_labels):  
        coords = np.where(labels_im == i)
        ymin, xmin = np.min(coords[0]), np.min(coords[1])
        ymax, xmax = np.max(coords[0]), np.max(coords[1])
        
        bounding_boxes.append((ymin, xmin, ymax, xmax))
    
    # Did exploratory data analysis on training data and came up with the below logic based on areas
    def calculate_area(coord):
        x1, y1, x2, y2 = coord
        return abs(x2 - x1) * abs(y2 - y1)
    
    areas = [calculate_area(coord) for coord in bounding_boxes]
    max_area = max(areas)
    
    filtered_bounding_boxes = []
    
    for each_index, each_area in enumerate(areas):
        if each_area/max_area >= 0.68:
            filtered_bounding_boxes.append(bounding_boxes[each_index])
            
    return filtered_bounding_boxes


def find_RBC(image):
    
    """
    To increase the accuracy, trained an image classifier using svm and got 99% accuracy for RBC.
    
    File name: BC_Classifier.py
    Saved Models:
        Features: feature_reduction.pkl
        Model: image_classifier.pkl
        
    To take the advantage of the image classifer, I tried to improve the object detection 
    but unable to detect most of them.
    """
    
    segments = slic(image, n_segments=200, compactness=10, start_label=0)
    cnt = len(np.unique(segments))
    
    group_means = np.zeros((cnt, 3), dtype="float32")
    for specific_group in range(cnt):
        mask_image = np.where(segments == specific_group, 255, 0).astype("uint8")
        mask_image = np.expand_dims(mask_image, axis=2)
        group_means[specific_group] = cv2.mean(image, mask=mask_image)[0:3]

    pixels = group_means.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 4  
    _, bestLabels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    target_color = np.array([125, 142, 175], dtype=np.float32)
    closest_group = np.argmin(np.linalg.norm(centers - target_color, axis=1))
    
    centers[:] = 0
    centers[closest_group] = (255, 255, 255)
    centers = centers.astype("uint8")
    colors_per_clump = centers[bestLabels.flatten()]
    
    cell_mask = colors_per_clump[segments]
    cell_mask = cv2.cvtColor(cell_mask, cv2.COLOR_BGR2GRAY)

    retval, labels = cv2.connectedComponents(cell_mask)

    bounding_boxes = []
    for i in range(1, retval):
        coords = np.where(labels == i)
        if coords[0].size > 0 and coords[1].size > 0:
            ymin, xmin = np.min(coords[0]), np.min(coords[1])
            ymax, xmax = np.max(coords[0]), np.max(coords[1])
            cropped_image = crop_image(image, xmin, xmax, ymin, ymax)
            
            sample_grayscale_image = preprocess(cropped_image)
            resized_cell = resize_with_padding(sample_grayscale_image, 64)
            features = pca_fit.transform([resized_cell.flatten()])
                        
            predicted_cell_type = model.predict(features)[0]
            if predicted_cell_type == "RBC":
            
                bounding_boxes.append((ymin, xmin, ymax, xmax))
    
    return bounding_boxes
