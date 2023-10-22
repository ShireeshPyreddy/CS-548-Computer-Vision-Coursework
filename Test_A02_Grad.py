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
import A02
from Test_A02 import *

marr_hildreth_params = [
    {"scale":7, "thresh":5},
    {"scale":3, "thresh":5},
    {"scale":15, "thresh":3},
]

class Test_A02_Grad(Test_A02):   
    def do_test_one_marr_hildreth_one_image(self, param_index, image_index):        
        # Get parameters
        mhp = marr_hildreth_params[param_index]
        
        # Load image
        image = cv2.imread(os.path.join(image_dir, self.all_filenames[image_index]),
                           cv2.IMREAD_GRAYSCALE)       
                
        # Do Marr-Hildreth
        output = A02.get_marr_hildreth_edges(image, **mhp)
                
        # Load ground image
        ground = cv2.imread(os.path.join(ground_dir, 
                                         ("MH_S_%02d_T_%02d" % (mhp["scale"], mhp["thresh"])), 
                                         self.all_filenames[image_index]),
                           cv2.IMREAD_GRAYSCALE)       
        
        # Compare to data
        GT.check_for_unequal("Failed on Marr-Hildreth (scale=" 
                             + str(mhp["scale"]) + ", thresh=" 
                             + str(mhp["thresh"]) + ") with image", 
                             self.all_filenames[image_index], 
                             output, ground)
        
    def test_marr_hildreth_image_0_0(self): self.do_test_one_marr_hildreth_one_image(0, 0)
    def test_marr_hildreth_image_0_1(self): self.do_test_one_marr_hildreth_one_image(0, 1)
    def test_marr_hildreth_image_0_2(self): self.do_test_one_marr_hildreth_one_image(0, 2)
    def test_marr_hildreth_image_0_3(self): self.do_test_one_marr_hildreth_one_image(0, 3)
    def test_marr_hildreth_image_0_4(self): self.do_test_one_marr_hildreth_one_image(0, 4)
    def test_marr_hildreth_image_0_5(self): self.do_test_one_marr_hildreth_one_image(0, 5)
    def test_marr_hildreth_image_0_6(self): self.do_test_one_marr_hildreth_one_image(0, 6)
    def test_marr_hildreth_image_0_7(self): self.do_test_one_marr_hildreth_one_image(0, 7)
    
    def test_marr_hildreth_image_1_0(self): self.do_test_one_marr_hildreth_one_image(1, 0)
    def test_marr_hildreth_image_1_1(self): self.do_test_one_marr_hildreth_one_image(1, 1)
    def test_marr_hildreth_image_1_2(self): self.do_test_one_marr_hildreth_one_image(1, 2)
    def test_marr_hildreth_image_1_3(self): self.do_test_one_marr_hildreth_one_image(1, 3)
    def test_marr_hildreth_image_1_4(self): self.do_test_one_marr_hildreth_one_image(1, 4)
    def test_marr_hildreth_image_1_5(self): self.do_test_one_marr_hildreth_one_image(1, 5)
    def test_marr_hildreth_image_1_6(self): self.do_test_one_marr_hildreth_one_image(1, 6)
    def test_marr_hildreth_image_1_7(self): self.do_test_one_marr_hildreth_one_image(1, 7)

    def test_marr_hildreth_image_2_0(self): self.do_test_one_marr_hildreth_one_image(2, 0)
    def test_marr_hildreth_image_2_1(self): self.do_test_one_marr_hildreth_one_image(2, 1)
    def test_marr_hildreth_image_2_2(self): self.do_test_one_marr_hildreth_one_image(2, 2)
    def test_marr_hildreth_image_2_3(self): self.do_test_one_marr_hildreth_one_image(2, 3)
    def test_marr_hildreth_image_2_4(self): self.do_test_one_marr_hildreth_one_image(2, 4)
    def test_marr_hildreth_image_2_5(self): self.do_test_one_marr_hildreth_one_image(2, 5)
    def test_marr_hildreth_image_2_6(self): self.do_test_one_marr_hildreth_one_image(2, 6)
    def test_marr_hildreth_image_2_7(self): self.do_test_one_marr_hildreth_one_image(2, 7)
    
def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A02_Grad))

if __name__ == '__main__':    
    main()
