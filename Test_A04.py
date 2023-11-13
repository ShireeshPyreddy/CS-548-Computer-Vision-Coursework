import os
import subprocess as sub
import cv2
import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch
import General_Testing as GT
from General_A04 import *
import A04

class Test_A04(unittest.TestCase):   

    @classmethod
    def setUpClass(cls):        
        # Load up original and ground truth data first...        
        cls.inputImages = []
        cls.groundImages = {}
        cls.groundFeatures = {}

        for i in range(len(inputImageFilenames)):
            image = cv2.imread(os.path.join(image_dir, inputImageFilenames[i]), cv2.IMREAD_GRAYSCALE)
            cls.inputImages.append(image)

        for label_type in LBP_LABEL_TYPES:
            cls.groundImages[label_type] = []
            for i in range(len(inputImageFilenames)):   
                lbp_filename = "LBP_" + label_type.value + "_" + inputImageFilenames[i]         
                gimage = cv2.imread(os.path.join(ground_dir, lbp_filename), cv2.IMREAD_GRAYSCALE)
                cls.groundImages[label_type].append(gimage)
          
        for label_type in LBP_LABEL_TYPES:
            cls.groundFeatures[label_type] = []  
            for region_cnt in range(1, 5):
                hist_filename = "HIST_" + label_type.value + "_" + str(region_cnt) + ".csv"                
                gdata = pd.read_csv(os.path.join(ground_dir, hist_filename))                
                cls.groundFeatures[label_type].append(gdata)
            
        cls.labeldata = pd.read_csv(os.path.join(ground_dir, "LABELS.csv"), index_col=0)
              
    ###########################################################################      
    # getOneLBPLabel   
    ###########################################################################    
      
    def do_test_getOneLBPLabel(self, label_type):
        # For each row...        
        for index, row in self.labeldata.iterrows():
            with self.subTest(csv_row=index):
                # Get subimage            
                subimage = row.iloc[3:].to_numpy()
                subimage = np.reshape(subimage, (3,3))
                # Get desired label
                true_label = row[label_type.value]
                # Test label
                pred_label = A04.getOneLBPLabel(subimage, label_type)    
                self.assertEqual(pred_label, true_label, "Label_Type: " + label_type.value + ", Subimage:\n" + str(subimage))
                
    def test_getOneLBPLabel_uniform(self):
        self.do_test_getOneLBPLabel(label_type=LBP_LABEL_TYPES.UNIFORM)
             
    ###########################################################################      
    # getLBPImage   
    ###########################################################################     
       
    def do_test_one_getLBPImage(self, index, label_type):
        # Load up original and ground truth images
        image = self.inputImages[index] 
        ground = self.groundImages[label_type][index] 
        # Compute LBP image
        lbp = A04.getLBPImage(image, label_type)        
        # Is it correct?        
        GT.check_for_unequal("Label_type: " + label_type.value, 
                             inputImageFilenames[index], lbp, ground)
    
    def do_test_getLBPImage(self, label_type):
        for image_index in range(len(inputImageFilenames)):            
            with self.subTest(image_index=image_index):
                self.do_test_one_getLBPImage(image_index, label_type)
                
    def test_getLBPImage_uniform(self):
        self.do_test_getLBPImage(label_type=LBP_LABEL_TYPES.UNIFORM)
        
    ###########################################################################      
    # getOneRegionLBPFeatures   
    ###########################################################################   
    
    def do_one_test_getOneRegionLBPFeatures(self, index, label_type):
        # Grab ground truth data for ONE big region
        gor = self.groundFeatures[label_type][0] 
        
        # Get ground LBP image
        groundLBP = self.groundImages[label_type][index]
            
        # Compute feature vector with image as one big region
        features = A04.getOneRegionLBPFeatures(groundLBP, label_type)
        
        # Grab the ground truth data for ONE big region for this particular image        
        oneGround = gor.loc[gor['Filename'] == inputImageFilenames[index]]            
        
        # Drop filename column        
        oneGround = oneGround.drop(columns=["Filename"])            
        
        # Convert to numpy array
        oneGround = oneGround.to_numpy()[0]            
        
        # Actually do test  
        GT.check_for_unequal("Label_type: " + label_type.value, 
                             inputImageFilenames[index], features, oneGround)
               
    
    def do_test_getOneRegionLBPFeatures(self, label_type):
        for image_index in range(len(inputImageFilenames)):
            with self.subTest(image_index=image_index):
                self.do_one_test_getOneRegionLBPFeatures(image_index, label_type)
                
    def test_getOneRegionLBPFeatures_uniform(self):
        self.do_test_getOneRegionLBPFeatures(label_type=LBP_LABEL_TYPES.UNIFORM)
        
    ###########################################################################      
    # getLBPFeatures   
    ###########################################################################   
    
    # Test each getLBPFeatures    
    def do_one_test_getLBPFeatures(self, regionSideCnt, label_type):
        # Get the index
        index = regionSideCnt - 1

        # Get appropriate ground truth data
        gor = self.groundFeatures[label_type][index]
        
        for image_index in range(len(self.groundImages[label_type])):
            with self.subTest(image_index=image_index):
            
                # Get ground truth image
                groundLBP = self.groundImages[label_type][image_index]    
        
                # Compute full feature vector...
                features = A04.getLBPFeatures(groundLBP, regionSideCnt, label_type)
                
                # Grab the ground truth data for this particular image
                oneGround = gor.loc[gor['Filename'] == inputImageFilenames[image_index]]            
                
                # Drop filename column
                oneGround = oneGround.drop(columns=["Filename"])            
                
                # Convert to numpy array
                oneGround = oneGround.to_numpy()[0]            
                
                # Actually do test  
                GT.check_for_unequal("Label_type: " + label_type.value, 
                             inputImageFilenames[index], features, oneGround)    
                        
    
    def do_test_getLBPFeatures(self, label_type):
        for rcnt in range(1, 5):
            with self.subTest(rcnt=rcnt):
                self.do_one_test_getLBPFeatures(rcnt, label_type)
                
    def test_getLBPFeatures_uniform(self):
        self.do_test_getLBPFeatures(label_type=LBP_LABEL_TYPES.UNIFORM)   
   
def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A04))

if __name__ == '__main__':    
    main()
