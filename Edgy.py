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
import math as m
from enum import Enum

class FilterType(Enum):
    BOX = 0
    GAUSS = 1
    MEDIAN = 2
    LAPLACE = 3
    LAPLACE_SHARP = 4
    SOBEL_X = 5
    SOBEL_Y = 6
    GRAD_MAG = 7
    CUSTOM = 8

def filter(image, filterSize, filterType, kernel=None):
    # output = np.copy(image)
    if filterType == FilterType.BOX:
        output = cv2.boxFilter(image, -1, (filterSize, filterSize))
    elif filterType == FilterType.GAUSS:
        output = cv2.GaussianBlur(image, (filterSize, 1), 0)
    elif filterType == FilterType.MEDIAN:
        output = cv2.medianBlur(image, filterSize)
    elif filterType == FilterType.LAPLACE:
        laplace = cv2.Laplacian(image, cv2.CV_32F, 
                               ksize=filterSize, 
                               scale=0.25)
        output = cv2.convertScaleAbs(laplace, alpha=0.5, beta=127.0)
    elif filterType == FilterType.LAPLACE_SHARP:
        laplace = cv2.Laplacian(image, cv2.CV_32F, 
                               ksize=filterSize, 
                               scale=0.25)
        fimage = image.astype("float32")
        fimage -= laplace
        output = cv2.convertScaleAbs(fimage)
    elif filterType == FilterType.SOBEL_X:
        sx = cv2.Sobel(image, 
                       cv2.CV_32F, 
                       1, 0, 
                       ksize=filterSize, 
                       scale=0.25)
        output = cv2.convertScaleAbs(sx, alpha=0.5, beta=127.0)
    elif filterType == FilterType.SOBEL_Y:
        sy = cv2.Sobel(image, 
                       cv2.CV_32F, 
                       0, 1, 
                       ksize=filterSize, 
                       scale=0.25)
        output = cv2.convertScaleAbs(sy, alpha=0.5, beta=127.0)
    elif filterType == FilterType.GRAD_MAG:
        sx = cv2.Sobel(image, 
                       cv2.CV_32F, 
                       1, 0, 
                       ksize=filterSize, 
                       scale=0.25)
        sy = cv2.Sobel(image, 
                       cv2.CV_32F, 
                       0, 1, 
                       ksize=filterSize, 
                       scale=0.25)
        grad_image = np.absolute(sx) + np.absolute(sy)
        output = cv2.convertScaleAbs(grad_image)
    elif filterType == FilterType.CUSTOM:
        if kernel is None:
            raise ValueError("Cannot use custom filter with None!")
        
        displayScale = np.sum(np.absolute(kernel))
        result = cv2.filter2D(image, cv2.CV_32F, kernel)
        output = cv2.convertScaleAbs(result, 
                                     alpha=1.0/displayScale,
                                     beta=127.0)
    else:
        output = np.copy(image)
    
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
        camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW recommended on Windows 
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
        
        filterSize = 27 #301 #27 #5
        filterType = FilterType.CUSTOM # FilterType.BOX
        
        kernel = np.zeros((filterSize, filterSize))
        scale = (10.0*m.pi)/filterSize
        for i in range(filterSize):
            kernel[:,i] = m.sin(i*scale)
            
        lowT = 100
        highT = 200
        
        # While not closed...
        ESC_KEY = 27
        key = -1
        while key != ESC_KEY:
            # Get next frame from camera
            _, frame = camera.read()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # output = filter(gray, filterSize, filterType, kernel)
            output = cv2.Canny(gray, lowT, highT)
                        
            # Show the image
            cv2.imshow(windowName, gray) #frame)
            cv2.imshow("FILTERED", output)

            # Wait 30 milliseconds, and grab any key presses
            key = cv2.waitKey(30)
            
            if key == ord('a'): lowT -= 10
            if key == ord('d'): lowT += 10
            if key == ord('w'): highT += 10
            if key == ord('s'): highT -= 10
            
            print(lowT, highT)

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