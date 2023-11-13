# MIT LICENSE
#
# Copyright 2023 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

import sys
import numpy as np
import torch
import tensorflow as tf
import cv2
import pandas
import sklearn
from enum import Enum

class ThreshType(Enum):
    BASIC = 0
    OTSU = 1
    COLOR = 2
    KMEANS = 3
    KMEANS_THRESH = 4
    

def do_segment(image, threshType, value, center=None):
    if threshType == ThreshType.BASIC:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value, output = cv2.threshold(image, value, 255, 
                               cv2.THRESH_BINARY)
    elif threshType == ThreshType.OTSU:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value, output = cv2.threshold(image, value, 255,
                                      cv2.THRESH_OTSU)
    elif threshType == ThreshType.COLOR:
        image = image.astype("float32")
        image = image[:,:] - center
        image = image * image
        #print("BEFORE:", image.shape)
        image = np.sum(image, axis=-1)
        #print("AFTER:", image.shape)     
        image = np.sqrt(image)  
        image /= np.sqrt(3) # Scale to [0,255]
        image = cv2.convertScaleAbs(image)
        value, output = cv2.threshold(image, value, 255, 
                               cv2.THRESH_BINARY_INV)
    elif threshType == ThreshType.KMEANS:
        image_shape = image.shape
        image = np.reshape(image, (-1, 3)).astype("float32")
        value, bestLabels, centers = cv2.kmeans(image,
                                                K=5,
                                                bestLabels=None,
                                                criteria=(
                                                   cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                   10, 1.0 
                                                ),
                                                attempts=10,
                                                flags=cv2.KMEANS_RANDOM_CENTERS)
        print(bestLabels.shape)
        print(centers.shape)
        centers = np.uint8(centers)
        output = centers[bestLabels.flatten()]
        print(output.shape)
        output = np.reshape(output, image_shape)
    elif threshType == ThreshType.KMEANS_THRESH:
        # Otsu's method
        orig_shape = image.shape
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        value, output = cv2.threshold(image, value, 255,
                                      cv2.THRESH_OTSU)
        # Grab foreground pixels
        foreground = np.where(output == 255)
        background = np.where(output != 255)
                
        def convert_to_coords(data):
            y, x = data            
            coords = np.stack([y,x], axis=1)
            coords = coords.astype("float32")
            return coords
            
        fore_coords = convert_to_coords(foreground)
        back_coords = convert_to_coords(background)
        
        print("Foreground:", fore_coords.shape)
        print("Background:", back_coords.shape)
        print("TOTAL:", (fore_coords.shape[0] + back_coords.shape[0]))
        
        #print(coords)     
        num_groups = 5
        value, bestLabels, centers = cv2.kmeans(fore_coords,
                                                K=num_groups,
                                                bestLabels=None,
                                                criteria=(
                                                   cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                                                   10, 1.0 
                                                ),
                                                attempts=10,
                                                flags=cv2.KMEANS_RANDOM_CENTERS)   
        print("Best labels:", bestLabels.shape)
        back_labels = np.zeros((back_coords.shape[0],))     
        back_labels[:] = num_groups        
        back_labels = np.reshape(back_labels, [-1, 1])
        print("Back labels:", back_labels.shape)
        all_labels = np.concatenate([bestLabels, back_labels], axis=0)
        print("All labels:", all_labels.shape)
        all_coords = np.concatenate([fore_coords, back_coords], axis=0)
        print("All coords:", all_coords.shape)
        
        centers = [
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,0],
            [0,255,255],
            [0,0,0]
        ]        
        centers = np.uint8(centers)   
        all_labels = np.uint8(all_labels)     
        colors = centers[all_labels.flatten()]
        
        print("Colors:", colors.shape)
        
        #colors = np.reshape(colors, orig_shape)
        #print("Colors AFTER:", colors.shape)
        
        output = np.zeros(orig_shape, dtype="uint8")
        print("OUTPUT SHAPE:", orig_shape)
        all_coords = np.uint8(all_coords)
        for c in range(len(all_coords)):
            coord = all_coords[c]
            #print(coord, colors[c])
            output[all_coords[c][0], all_coords[c][1]] = colors[c]  
            #print(output[all_coords[c][0], all_coords[c][1]])     
        print("OUTPUT SHAPE after:", orig_shape)
    return value, output 

def get_connected_image(thresh_image):
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    thresh_image = cv2.morphologyEx(thresh_image, cv2.MORPH_OPEN,
                                    element,iterations=1)
        
    centers = [
            [255,0,0],
            [0,255,0],
            [0,0,255],
            [255,255,0],
            [0,255,255],
            [255,255,255]
        ]     
    
    num_components, label_image = cv2.connectedComponents(thresh_image,
                                                          connectivity=8,
                                                          ltype=cv2.CV_32S)
    print("Number of connected components:", num_components)
    
    output = np.zeros(thresh_image.shape + (3,), dtype="uint8")
    
    for row in range(output.shape[0]):
        for col in range(output.shape[1]):
            label = label_image[row,col]
            if label > 0:
                label -= 1
                label %= len(centers)
                output[row,col] = centers[label]
                
    for label in range(num_components):
        label += 1
        coords = np.where(label_image == label)
        print(coords)
                
    return output     

###############################################################################
# MAIN
###############################################################################

def main():
    ###############################################################################
    # TENSORFLOW
    ###############################################################################

    a = tf.constant("Hello Tensorflow!")
    tf.print(a)
    print(tf.config.list_physical_devices('GPU'))           # Should list GPU devices
    print(tf.reduce_sum(tf.random.normal([1000, 1000])))    # Should print number tensor
    
    ###############################################################################
    # PYTORCH
    ###############################################################################
    
    b = torch.rand(5,3)
    print(b)
    print("Torch CUDA?:", torch.cuda.is_available())
    
    ###############################################################################
    # PRINT OUT VERSIONS
    ###############################################################################

    print("Tensorflow:", tf.__version__)    
    print("Torch:", torch.__version__)
    print("Numpy:", np.__version__)
    print("OpenCV:", cv2.__version__)
    print("Pandas:", pandas.__version__)
    print("Scikit-Learn:", sklearn.__version__)
        
    ###############################################################################
    # OPENCV
    ###############################################################################
    if len(sys.argv) <= 1:
        # Webcam
        print("Opening webcam...")

        # Linux/Mac (or native Windows) with direct webcam connection
        camera = cv2.VideoCapture(1, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
        # WSL: Use Yawcam to stream webcam on webserver
        # https://www.yawcam.com/download.php
        # Get local IP address and replace
        #IP_ADDRESS = "192.168.0.7"    
        #camera = cv2.VideoCapture("http://" + IP_ADDRESS + ":8081/video.mjpg")
        
        # Did we get it?
        if not camera.isOpened():
            print("ERROR: Cannot open camera!")
            exit(1)

        # Create window ahead of time
        windowName = "Webcam"
        cv2.namedWindow(windowName)
        
        value = 128
        center = (0,0,128)

        # While not closed...
        key = -1
        ESC_KEY = 27
        while key != ESC_KEY:
            # Get next frame from camera
            _, frame = camera.read()
                        
            #value, output = do_segment(frame, ThreshType.COLOR, value,
            #                           center)
            '''
            value, blue_output = do_segment(frame, ThreshType.COLOR, value,
                                       center=(128,0,0))
            _, green_output = do_segment(frame, ThreshType.COLOR, value,
                                       center=(0,128,0))
            _, red_output = do_segment(frame, ThreshType.COLOR, value,
                                       center=(0,0,128))
            '''
            #value, output = do_segment(frame, ThreshType.KMEANS_THRESH, value,
            #                           center)
            
            value, output = do_segment(frame, ThreshType.OTSU, value,
                                       center)
            
            vis_output = get_connected_image(output)
                        
            # Show the image
            cv2.imshow(windowName, frame)
            cv2.imshow("SEGMENT", output) 
            cv2.imshow("GROUPS", vis_output)
            #cv2.imshow("RED", red_output)
            #cv2.imshow("GREEN", green_output)
            #cv2.imshow("BLUE", blue_output)          

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
                        
            if key == ord('a'): value -= 10
            if key == ord('d'): value += 10
            print("VALUE:", value)

        # Release the camera and destroy the window
        camera.release()
        cv2.destroyAllWindows()

        # Close down...
        print("Closing application...")

    else:
        # Trying to load image from argument

        # Get filename
        filename = sys.argv[1]

        # Load image
        print("Loading image:", filename)
        image = cv2.imread(filename) # For grayscale: cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        # Check if data is invalid
        if image is None:
            print("ERROR: Could not open or find the image!")
            exit(1)

        # Show our image (with the filename as the window title)
        windowTitle = "PYTHON: " + filename
        cv2.imshow(windowTitle, image)

        # Wait for a keystroke to close the window
        cv2.waitKey(-1)

        # Cleanup this window
        cv2.destroyAllWindows()

if __name__ == "__main__": 
    main()