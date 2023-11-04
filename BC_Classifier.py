import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import joblib
import cv2
import tensorflow_datasets as tfds

from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, classification_report

import tensorflow as tf

#reproducability!
random_seed = 27
import random
random.seed(random_seed)
np.random.seed(random_seed)

def load_and_prepare_BCCD_data():
    # Fragments taken from: https://www.tensorflow.org/tutorials/images/segmentation

    # Load the BCCD dataset: https://www.tensorflow.org/datasets/catalog/bccd
    dataset, info = tfds.load('bccd', with_info=True)

    # Create mapping function to process each datapoint    
    def prepare_datapoint(datapoint):
        input_image = datapoint['image']
        input_objects = datapoint['objects']        
        return input_image, input_objects

    # Create TF datasets for training and testing
    train_data = dataset['train'].map(prepare_datapoint, num_parallel_calls=tf.data.AUTOTUNE)
    val_data = dataset['validation'].map(prepare_datapoint, num_parallel_calls=tf.data.AUTOTUNE)
    test_data = dataset['test'].map(prepare_datapoint, num_parallel_calls=tf.data.AUTOTUNE)

    # Number of items
    print("Number of training images:", info.splits['train'].num_examples)
    print("Number of val images:", info.splits['validation'].num_examples)
    print("Number of testing images:", info.splits['test'].num_examples)  

    return train_data, val_data, test_data

def show_image_with_bounding_boxes(image, image_title, bounding_boxes, labels, shape=None):
    fig,ax = plt.subplots(1, figsize=(10, 10))
    clone = image.copy()
    if(bounding_boxes is not None):
        for lab, bb in zip(labels, bounding_boxes):
            # ymin, xmin, ymax, xmax = box
            # print(lab, bb, (bb[1], bb[0]), (bb[3], bb[2]), (int(bb[1]*shape[1]), int(bb[0])*shape[0]),
                #   (int(bb[3])*shape[1], int(bb[2])*shape[0]))
            
            if type(lab) is int:
                label = "RBC" if lab == 0 else "WBC" if lab == 1 else "Plate" 
            else:
                label = lab
            
            if label == "RBC":
            
                if shape is not None:
                    clone = cv2.rectangle(clone,
                                        (int(bb[1]*shape[1]), int(bb[0]*shape[0])),
                                        (int(bb[3]*shape[1]), int(bb[2]*shape[0])),
                                        (0, 0, 255),
                                        2)
                
                    cv2.putText(clone,
                                label, 
                                (int(bb[1]*shape[1]), int(bb[0]*shape[0])),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 255),
                                2)
                else:
                    clone = cv2.rectangle(clone,
                                        (bb[1], bb[0]),
                                        (bb[3], bb[2]),
                                        (0, 0, 255),
                                        2)
                
                    cv2.putText(clone,
                                label, 
                                (bb[1], bb[0]),
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (0, 0, 255),
                                2)
    
    ax.imshow(clone)
    ax.set_title(image_title)
    ax.axis('off')
    plt.show()

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


def increase_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    l2 = clahe.apply(l)

    lab = cv2.merge((l2,a,b))
    img2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR) 
    
    return img2

def preprocess(image):
    image_with_increased_contrast = increase_contrast(image)

    gray_image = cv2.cvtColor(image_with_increased_contrast, cv2.COLOR_BGR2GRAY)
  
    return gray_image


def find_cell_edges(grayscale_image):
    smoothen_image = cv2.GaussianBlur(grayscale_image, (3, 3), 0)

    otsu_threshold, otsu_image = cv2.threshold(smoothen_image, 0, 255,
                                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    image_closed_otsu = cv2.morphologyEx(otsu_image, cv2.MORPH_CLOSE, 
                                         structuring_element, iterations = 3)
    
    canny_image = cv2.Canny(image_closed_otsu, otsu_threshold, 0.1*otsu_threshold)
    
    return canny_image

class CellImage:
    
    def __init__(self, image, base_file_name):
        self.original_image = image 
        self.base_file_name = base_file_name

    def resize(self, dimension):
        processed_image = preprocess(self.original_image)
        resized_image = resize_with_padding(processed_image, dimension)
        return resized_image

    def size(self):
        return self.original_image.shape
    
def crop_image(image, xmin, xmax, ymin, ymax):
    # check for incorrect annotations of the bounding box
    if(ymax - ymin > 0 and xmax - xmin > 0):
        cropped_image = image[ymin:ymax, xmin:xmax]
        return cropped_image
    else: 
        return None

def get_cell_image_by_bounding_box(image, bounding_box, dimension, image_shape):
    ymin, xmin, ymax, xmax = bounding_box
    """_summary_
    (int(bb[1]*shape[1]), int(bb[0]*shape[0])),
                                  (int(bb[3]*shape[1]), int(bb[2]*shape[0])),
    """
    return crop_image(image, 
                      int(xmin*image_shape[1]), 
                      int(xmax*image_shape[1]), 
                      int(ymin*image_shape[0]),
                      int(ymax*image_shape[0]))
    
def extract_image_samples(data):
    
    min_index, max_index = 0, 411
    dimension = 64
    
    images_count = max_index - min_index
    images = np.array([])
    labels = np.array([])
    
    for i, sample in data:
        image = i
        image = cv2.cvtColor(image.numpy(), cv2.COLOR_RGB2BGR)
        label = tfds.as_numpy(sample['label'])
        bbox = tfds.as_numpy(sample['bbox'])
        # show_image_with_bounding_boxes(image, "Cell types", bbox, label, image.shape)
        # break

        for bounding_box, lab in zip(bbox, label):
            label_named = "RBC" if lab == 0 else "WBC" if lab == 1 else "Plate" 
            cropped_cell = get_cell_image_by_bounding_box(image, bounding_box, dimension, image.shape)

            if(cropped_cell is not None):
                images = np.append(images, CellImage(cropped_cell, "BCCD"))
                labels = np.append(labels, label_named)

    return images, labels


def create_df(images, labels):
    data = []
    for cell_image, cell_label in zip(images, labels):
        img = cell_image.resize(64)
        w, h, _ = cell_image.original_image.shape
        bgr = cv2.mean(cell_image.original_image)
        
        data.append([np.float32(img.flatten()), w, h, bgr[2], bgr[1], bgr[0], cell_label])

    cells_data = pd.DataFrame(data, columns=["image_vector",
                                            "bounding_box_width", 
                                            "bounding_box_height", 
                                            "mean_red_color_intensity",
                                            "mean_blue_color_intesity", 
                                            "mean_green_color_intensity",
                                            "cell_type"])
    return cells_data

train, val, test = load_and_prepare_BCCD_data()

start_time = time.time()

train_cell_images, train_cell_labels = extract_image_samples(train)
val_cell_images, val_cell_labels = extract_image_samples(val)
test_cell_images, test_cell_labels = extract_image_samples(test)

print("Train Images shape:", train_cell_images.shape)
print("Train Labels shape:", train_cell_labels.shape)

print("Val Images shape:", val_cell_images.shape)
print("Val Labels shape:", val_cell_labels.shape)

print("Test Images shape:", test_cell_images.shape)
print("Test Labels shape:", test_cell_labels.shape)

end_time = time.time()
elapsed_time = end_time - start_time
print("Total elapsed time:", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


train_df = create_df(train_cell_images, train_cell_labels)
val_df = create_df(val_cell_images, val_cell_labels)
test_df = create_df(test_cell_images, test_cell_labels)

print(train_df.head())

combined_df = pd.concat([train_df, val_df], ignore_index=True)

print(combined_df.shape)

images_pca = PCA(n_components=100, random_state=random_seed)
pca_fit = images_pca.fit(combined_df['image_vector'].tolist())
pcas = pca_fit.transform(combined_df['image_vector'].tolist())
test_pcas = pca_fit.transform(test_df['image_vector'].tolist())

print("Each cell image PCAs shape:", pcas[0].shape)

model = SVC(kernel='rbf',
          C = 1, #default 
          gamma="scale", # default = 1 / (n_features * X.var())
          class_weight="balanced",
          decision_function_shape="ovr", #only option, ovo deprecated
          random_state=random_seed,
          probability=True)

svc_model = model.fit(pcas, combined_df['cell_type'].tolist())
print(svc_model.score(pcas, combined_df['cell_type'].tolist()))

print(svc_model.score(test_pcas, test_df['cell_type'].tolist()))

print("Classification report for classifier", classification_report(test_df['cell_type'].tolist(), 
                                                                        svc_model.predict(test_pcas)))


# joblib.dump(pca_fit, 'feature_reduction.pkl')
# joblib.dump(lr_res, 'image_classifier.pkl')
