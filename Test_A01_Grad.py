import unittest
from unittest.mock import patch
import shutil
from pathlib import Path

import sys
import os
import subprocess as sub
import cv2
import numpy as np
import pandas as pd
import General_Testing as GT
import A01
from Test_A01 import *

class Test_A01_Grad(Test_A01):   
    def test_get_hist_equalize_transform_with_cl(self):
        # For each image...
        all_filenames = os.listdir(image_dir)
        all_filenames.sort()
        
        # Load up CSVs
        df_eqs_200 = pd.read_csv(os.path.join(ground_dir,"data_transform_eqs_cl_200.csv"))
        df_eqs_40 = pd.read_csv(os.path.join(ground_dir,"data_transform_eqs_cl_40.csv"))
        
        # For each image
        for filename in all_filenames:
            # Load image
            image = cv2.imread(os.path.join(image_dir, filename))
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Call histogram equalization function
            output_eqs_200 = A01.get_hist_equalize_transform(image, True, do_cl=True, cl_thresh=200)
            output_eqs_40 = A01.get_hist_equalize_transform(image, True, do_cl=True, cl_thresh=40)
            # Load up ground truth transformations
            ground_eqs_200 = get_matching_hist(df_eqs_200, filename)
            ground_eqs_40 = get_matching_hist(df_eqs_40, filename)
                        
            # Compare to see if correct     
            GT.check_for_unequal("Failed on image (cl=200)", filename, output_eqs_200, ground_eqs_200)   
            GT.check_for_unequal("Failed on image (cl=40)", filename, output_eqs_40, ground_eqs_40)   
            
    def test_do_adaptive_histogram_equalize(self):
        # For each image...
        all_filenames = os.listdir(image_dir)
        all_filenames.sort()
        
        for filename in all_filenames:
            # Load image
            image = cv2.imread(os.path.join(image_dir, filename))
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Call histogram equalization function
            output_b3_cl40 = A01.do_adaptive_histogram_equalize(image, 3, 40)
            output_b8_cl40 = A01.do_adaptive_histogram_equalize(image, 8, 40)
            # Load up ground truth image            
            ground_b3_cl40 = cv2.imread(os.path.join(ground_dir, "CLAHE_3_40_" + filename))
            ground_b3_cl40 = cv2.cvtColor(ground_b3_cl40, cv2.COLOR_BGR2GRAY)
            
            ground_b8_cl40 = cv2.imread(os.path.join(ground_dir, "CLAHE_8_40_" + filename))            
            ground_b8_cl40 = cv2.cvtColor(ground_b8_cl40, cv2.COLOR_BGR2GRAY)
            
            # Compare to see if correct     
            GT.check_for_unequal("Failed on image (CLAHE,3,40)", filename, output_b3_cl40, ground_b3_cl40)   
            GT.check_for_unequal("Failed on image (CLAHE,8,40)", filename, output_b8_cl40, ground_b8_cl40)   
           
def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A01_Grad))

if __name__ == '__main__':    
    main()
