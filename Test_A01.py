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

base_dir = "assign01"
image_dir = base_dir + "/" + "images"
ground_dir = base_dir + "/" + "ground"
out_dir = base_dir + "/" + "output"

def get_matching_hist(df, filename):
    return df.loc[df['filename'] == filename].drop(["filename"], axis=1).to_numpy()[0]
    
class Test_A01(unittest.TestCase):   
    
    def test_create_unnormalized_hist(self):
        # Load unnormalized histogram list
        df = pd.read_csv(os.path.join(ground_dir,"data_unnorm_hist.csv"))
        # Get filenames
        all_filenames = df["filename"]
        # For each image...
        for filename in all_filenames:
            # Load image
            image = cv2.imread(os.path.join(image_dir, filename))
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Call hist function
            hist = A01.create_unnormalized_hist(image)
            # Compare to correct answer
            ground_hist = get_matching_hist(df, filename)
            GT.check_for_unequal("Failed on image", filename, hist, ground_hist)
                                
    def test_normalize_hist(self):
        # Load up CSVs
        input_df = pd.read_csv(os.path.join(ground_dir,"data_unnorm_hist.csv"))
        ground_df = pd.read_csv(os.path.join(ground_dir,"data_norm_hist.csv"))
        # Sort by filename
        input_df = input_df.sort_values("filename", axis=0)
        ground_df = ground_df.sort_values("filename", axis=0)
        # Get filename list
        all_filenames = input_df["filename"].to_numpy()
        # Drop filename column
        all_input_hist = input_df.drop(["filename"], axis=1)
        all_ground_hist = ground_df.drop(["filename"], axis=1)        
        # Loop through rows        
        for index, row in all_input_hist.iterrows():
            # Grab the input unnormalized histogram
            input_hist = row.to_numpy()
            # Get ground truth
            ground_hist = all_ground_hist.iloc[[index]].to_numpy()[0]            
            # Calculate the normalized histogram
            norm_hist = A01.normalize_hist(input_hist)
            # Compare to see if correct     
            GT.check_for_unequal("Failed on image", all_filenames[index], norm_hist, ground_hist)        
                    
    def test_create_cdf(self):
        # Load up CSVs
        input_df = pd.read_csv(os.path.join(ground_dir,"data_norm_hist.csv"))
        ground_df = pd.read_csv(os.path.join(ground_dir,"data_cdf.csv"))
        # Sort by filename
        input_df = input_df.sort_values("filename", axis=0)
        ground_df = ground_df.sort_values("filename", axis=0)
        # Get filename list
        all_filenames = input_df["filename"].to_numpy()
        # Drop filename column
        all_input_hist = input_df.drop(["filename"], axis=1)
        all_ground = ground_df.drop(["filename"], axis=1)        
        # Loop through rows        
        for index, row in all_input_hist.iterrows():
            # Grab the input normalized histogram
            input_hist = row.to_numpy()
            # Get ground truth
            ground = all_ground.iloc[[index]].to_numpy()[0]            
            # Calculate the cdf
            cdf = A01.create_cdf(input_hist)
            # Compare to see if correct      
            GT.check_for_unequal("Failed on image", all_filenames[index], cdf, ground)        
                         
    def test_get_hist_equalize_transform(self):
        # For each image...
        all_filenames = os.listdir(image_dir)
        all_filenames.sort()
        
        # Load up CSVs
        df_eq = pd.read_csv(os.path.join(ground_dir,"data_transform_eq.csv"))
        df_eqs = pd.read_csv(os.path.join(ground_dir,"data_transform_eqs.csv"))
        
        # For each image
        for filename in all_filenames:
            # Load image
            image = cv2.imread(os.path.join(image_dir, filename))
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Call histogram equalization function
            output_eq = A01.get_hist_equalize_transform(image, False)
            output_eqs = A01.get_hist_equalize_transform(image, True)
            # Load up ground truth transformations
            ground_eq = get_matching_hist(df_eq, filename)
            ground_eqs = get_matching_hist(df_eqs, filename)
                        
            # Compare to see if correct     
            GT.check_for_unequal("Failed on image (no stretch)", filename, output_eq, ground_eq)   
            GT.check_for_unequal("Failed on image (stretch)", filename, output_eqs, ground_eqs)   
            
    def test_do_histogram_equalize(self):
        # For each image...
        all_filenames = os.listdir(image_dir)
        all_filenames.sort()
        
        for filename in all_filenames:
            # Load image
            image = cv2.imread(os.path.join(image_dir, filename))
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Call histogram equalization function
            output_eq = A01.do_histogram_equalize(image, False)
            output_eqs = A01.do_histogram_equalize(image, True)
            # Load up ground truth image            
            ground_eq = cv2.imread(os.path.join(ground_dir, "EQ_" + filename))
            ground_eq = cv2.cvtColor(ground_eq, cv2.COLOR_BGR2GRAY)
            
            ground_eqs = cv2.imread(os.path.join(ground_dir, "EQS_" + filename))            
            ground_eqs = cv2.cvtColor(ground_eqs, cv2.COLOR_BGR2GRAY)
            
            # Compare to see if correct     
            GT.check_for_unequal("Failed on image (no stretch)", filename, output_eq, ground_eq)   
            GT.check_for_unequal("Failed on image (stretch)", filename, output_eqs, ground_eqs)   
           
def main():
    runner = unittest.TextTestRunner()
    runner.run(unittest.makeSuite(Test_A01))

if __name__ == '__main__':    
    main()
