###############################################################################
# IMPORTS
###############################################################################

import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from sklearn.metrics import (accuracy_score, f1_score)
import time
import A05

base_dir = "all_assign05"
out_dir = base_dir + "/" + "output"

###############################################################################
# MAIN
###############################################################################

def main():   
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Get names of all approaches
    all_names = A05.get_approach_names()
    chosen_approach_names = all_names
    
    for approach_name in chosen_approach_names:
        print("TRAINING APPROACH:", approach_name)    
            
        # Create data transforms
        train_transform = A05.get_data_transform(approach_name, training=True)
        test_transform = A05.get_data_transform(approach_name, training=False)
            
        # Load CIFAR10 data
        training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
        test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
        
        # Create dataloaders
        batch_size = A05.get_batch_size(approach_name)
        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size)
            
        # Set number of classes
        class_cnt = 10    
        
        # Create the model
        model = A05.create_model(approach_name, class_cnt)
        print("MODEL:", approach_name)
        print(model)
        
        # Move to GPU if possible
        device = ("cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {device} device")
            
        model = model.to(device)
                     
        # Train classifiers
        start_time = time.time()
        print("Training " + approach_name + "...")
        model = A05.train_model(approach_name, model, device, train_dataloader, test_dataloader)
        print("Training complete!")
        print("Time taken:", (time.time() - start_time))
        
        # Save the model
        model_path = os.path.join(out_dir, "model_" + approach_name + ".pth")
        torch.save(model.state_dict(), model_path)
        print("Model saved to:", model_path)
    
if __name__ == "__main__": 
    main()
    