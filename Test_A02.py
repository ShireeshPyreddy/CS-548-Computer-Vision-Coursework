import unittest
from unittest.mock import patch
import General_Testing as GT
import shutil
from pathlib import Path

import sys

import os
import subprocess as sub
import cv2
import numpy as np
import A02

base_dir = "assign02"
image_dir = base_dir + "/" + "images"
filter_dir = base_dir + "/" + "filters"
ground_dir = base_dir + "/" + "ground"
out_dir = base_dir + "/" + "output"

ground_kernels = [
        np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64),
        np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64),
        np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64),
        np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64),
        np.array([[1],[0],[-1]], dtype=np.float64),
        np.array([[1, 0, -1]], dtype=np.float64),
        np.array([[1, 2, 3, 4, 5],
                  [6, 7, 8, 9, 10], 
                  [11, 12, 13, 14, 15], 
                  [16, 17, 18, 19, 20], 
                  [21, 22, 23, 24, 25], 
                  [26, 27, 28, 29, 30], 
                  [31, 32, 33, 34, 35]], dtype=np.float64)
]

alphaBetaValues = [
    [0.125, 127],
    [0.125, 127],
    [0.0625, 0],
    [0.125, 127],
    [0.125, 127],
    [0.125, 127],
    [0.0015873015, 0]
]

class Test_A02(unittest.TestCase):
    def setUp(self):
        self.all_filenames = os.listdir(image_dir)
        self.all_filenames.sort()
    
        self.all_filters = os.listdir(filter_dir)
        self.all_filters.sort()
        
    def do_test_one_filter_load(self, findex):
        # Get filter filename
        filter_filename = self.all_filters[findex]
        # Load using function
        kernel = A02.read_kernel_file(os.path.join(filter_dir, filter_filename))
        # Get ground kernels
        ground = ground_kernels[findex]
        # Compare to data
        GT.check_for_unequal("Failed on filter", filter_filename, kernel, ground)
        
    def test_filter_load_0(self): self.do_test_one_filter_load(0)
    def test_filter_load_1(self): self.do_test_one_filter_load(1)
    def test_filter_load_2(self): self.do_test_one_filter_load(2)
    def test_filter_load_3(self): self.do_test_one_filter_load(3)
    def test_filter_load_4(self): self.do_test_one_filter_load(4)
    def test_filter_load_5(self): self.do_test_one_filter_load(5)
    def test_filter_load_6(self): self.do_test_one_filter_load(6)
    
    def do_test_one_filter_one_image(self, findex, image_index):        
        # Get ground kernels
        kernel = ground_kernels[findex]
        
        # Get kernel name
        kernel_name = self.all_filters[findex][:-4]
        
        # Load image
        image = cv2.imread(os.path.join(image_dir, self.all_filenames[image_index]),
                           cv2.IMREAD_GRAYSCALE)       
        
        # Get alpha/beta values
        alpha_beta = alphaBetaValues[findex]
        
        # Filter image
        output = A02.apply_filter(image, kernel=kernel, alpha=alpha_beta[0], beta=alpha_beta[1])
        
        # Load ground image
        ground = cv2.imread(os.path.join(ground_dir, kernel_name, self.all_filenames[image_index]),
                           cv2.IMREAD_GRAYSCALE)       
        
        # Compare to data
        GT.check_for_unequal("Failed on filter " + str(findex) + " with image", 
                             self.all_filenames[image_index], 
                             output, ground)
        
    def test_filter_image_0_0(self): self.do_test_one_filter_one_image(0, 0)
    def test_filter_image_0_1(self): self.do_test_one_filter_one_image(0, 1)
    def test_filter_image_0_2(self): self.do_test_one_filter_one_image(0, 2)
    def test_filter_image_0_3(self): self.do_test_one_filter_one_image(0, 3)
    def test_filter_image_0_4(self): self.do_test_one_filter_one_image(0, 4)
    def test_filter_image_0_5(self): self.do_test_one_filter_one_image(0, 5)
    def test_filter_image_0_6(self): self.do_test_one_filter_one_image(0, 6)
    def test_filter_image_0_7(self): self.do_test_one_filter_one_image(0, 7)
    
    def test_filter_image_1_0(self): self.do_test_one_filter_one_image(1, 0)
    def test_filter_image_1_1(self): self.do_test_one_filter_one_image(1, 1)
    def test_filter_image_1_2(self): self.do_test_one_filter_one_image(1, 2)
    def test_filter_image_1_3(self): self.do_test_one_filter_one_image(1, 3)
    def test_filter_image_1_4(self): self.do_test_one_filter_one_image(1, 4)
    def test_filter_image_1_5(self): self.do_test_one_filter_one_image(1, 5)
    def test_filter_image_1_6(self): self.do_test_one_filter_one_image(1, 6)
    def test_filter_image_1_7(self): self.do_test_one_filter_one_image(1, 7)
    
    def test_filter_image_2_0(self): self.do_test_one_filter_one_image(2, 0)
    def test_filter_image_2_1(self): self.do_test_one_filter_one_image(2, 1)
    def test_filter_image_2_2(self): self.do_test_one_filter_one_image(2, 2)
    def test_filter_image_2_3(self): self.do_test_one_filter_one_image(2, 3)
    def test_filter_image_2_4(self): self.do_test_one_filter_one_image(2, 4)
    def test_filter_image_2_5(self): self.do_test_one_filter_one_image(2, 5)
    def test_filter_image_2_6(self): self.do_test_one_filter_one_image(2, 6)
    def test_filter_image_2_7(self): self.do_test_one_filter_one_image(2, 7)
    
    def test_filter_image_3_0(self): self.do_test_one_filter_one_image(3, 0)
    def test_filter_image_3_1(self): self.do_test_one_filter_one_image(3, 1)
    def test_filter_image_3_2(self): self.do_test_one_filter_one_image(3, 2)
    def test_filter_image_3_3(self): self.do_test_one_filter_one_image(3, 3)
    def test_filter_image_3_4(self): self.do_test_one_filter_one_image(3, 4)
    def test_filter_image_3_5(self): self.do_test_one_filter_one_image(3, 5)
    def test_filter_image_3_6(self): self.do_test_one_filter_one_image(3, 6)
    def test_filter_image_3_7(self): self.do_test_one_filter_one_image(3, 7)
    
    def test_filter_image_4_0(self): self.do_test_one_filter_one_image(4, 0)
    def test_filter_image_4_1(self): self.do_test_one_filter_one_image(4, 1)
    def test_filter_image_4_2(self): self.do_test_one_filter_one_image(4, 2)
    def test_filter_image_4_3(self): self.do_test_one_filter_one_image(4, 3)
    def test_filter_image_4_4(self): self.do_test_one_filter_one_image(4, 4)
    def test_filter_image_4_5(self): self.do_test_one_filter_one_image(4, 5)
    def test_filter_image_4_6(self): self.do_test_one_filter_one_image(4, 6)
    def test_filter_image_4_7(self): self.do_test_one_filter_one_image(4, 7)
    
    def test_filter_image_5_0(self): self.do_test_one_filter_one_image(5, 0)
    def test_filter_image_5_1(self): self.do_test_one_filter_one_image(5, 1)
    def test_filter_image_5_2(self): self.do_test_one_filter_one_image(5, 2)
    def test_filter_image_5_3(self): self.do_test_one_filter_one_image(5, 3)
    def test_filter_image_5_4(self): self.do_test_one_filter_one_image(5, 4)
    def test_filter_image_5_5(self): self.do_test_one_filter_one_image(5, 5)
    def test_filter_image_5_6(self): self.do_test_one_filter_one_image(5, 6)
    def test_filter_image_5_7(self): self.do_test_one_filter_one_image(5, 7)
    
    def test_filter_image_6_0(self): self.do_test_one_filter_one_image(6, 0)
    def test_filter_image_6_1(self): self.do_test_one_filter_one_image(6, 1)
    def test_filter_image_6_2(self): self.do_test_one_filter_one_image(6, 2)
    def test_filter_image_6_3(self): self.do_test_one_filter_one_image(6, 3)
    def test_filter_image_6_4(self): self.do_test_one_filter_one_image(6, 4)
    def test_filter_image_6_5(self): self.do_test_one_filter_one_image(6, 5)
    def test_filter_image_6_6(self): self.do_test_one_filter_one_image(6, 6)
    def test_filter_image_6_7(self): self.do_test_one_filter_one_image(6, 7)
        
def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A02))

if __name__ == '__main__':    
    main()

