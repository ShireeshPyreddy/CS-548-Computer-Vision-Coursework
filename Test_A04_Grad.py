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
import A04
from Test_A04 import *

class Test_A04_Grad(Test_A04):  
    ###########################################################################      
    # getOneLBPLabel   
    ###########################################################################  
     
    def test_getOneLBPLabel_uniformrot(self):
        self.do_test_getOneLBPLabel(label_type=LBP_LABEL_TYPES.UNIFORM_ROT)
        
    def test_getOneLBPLabel_full(self):
        self.do_test_getOneLBPLabel(label_type=LBP_LABEL_TYPES.FULL)
        
    ###########################################################################      
    # getLBPImage   
    ###########################################################################   
        
    def test_getLBPImage_uniformrot(self):
        self.do_test_getLBPImage(label_type=LBP_LABEL_TYPES.UNIFORM_ROT)
        
    def test_getLBPImage_full(self):
        self.do_test_getLBPImage(label_type=LBP_LABEL_TYPES.FULL)
        
    ###########################################################################      
    # getOneRegionLBPFeatures   
    ###########################################################################   
    
    def test_getOneRegionLBPFeatures_uniformrot(self):
        self.do_test_getOneRegionLBPFeatures(label_type=LBP_LABEL_TYPES.UNIFORM_ROT)
        
    def test_getOneRegionLBPFeatures_full(self):
        self.do_test_getOneRegionLBPFeatures(label_type=LBP_LABEL_TYPES.FULL)
        
    ###########################################################################      
    # getLBPFeatures   
    ###########################################################################   
    
    def test_getLBPFeatures_uniformrot(self):
        self.do_test_getLBPFeatures(label_type=LBP_LABEL_TYPES.UNIFORM_ROT) 
        
    def test_getLBPFeatures_full(self):
        self.do_test_getLBPFeatures(label_type=LBP_LABEL_TYPES.FULL) 

def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A04_Grad))

if __name__ == '__main__':    
    main()
