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

base_dir = "assign05"
out_dir = base_dir + "/" + "output"

###############################################################################
# CALCULATE METRICS
###############################################################################

def compute_metrics(ground, pred):    
    scores = {}
    scores["accuracy"] = accuracy_score(y_true=ground, y_pred=pred)
    scores["f1"] = f1_score(y_true=ground, y_pred=pred, average="macro")
    return scores

###############################################################################
# GET PREDICTIONS FROM MODEL
###############################################################################

def get_predictions_and_ground(model, dataloader, device):
    # Set model to evaluation mode
    model.eval()
    # Create lists for ground and pred
    all_ground = []
    all_pred = []
    
    with torch.no_grad():
        for X, y in dataloader:
            # Append ground truth info
            all_ground.append(y)
            
            # Move data to device
            X, y = X.to(device), y.to(device)
            
            # Run prediction
            pred = model(X)
            
            # Get largest class prediction
            pred = pred.argmax(1)
            
            # Move to CPU
            pred = pred.cpu()
            
            # Append to list
            all_pred.append(pred)
            
    # Convert to single Tensor and then numpy
    all_ground = torch.concat(all_ground).numpy()
    all_pred = torch.concat(all_pred).numpy()
    
    return {"ground": all_ground, "pred": all_pred}
  
###############################################################################
# PRINTS RESULTS (to STDOUT or file)
###############################################################################
def print_results(approach_data, stream=sys.stdout):
    boundary = "****************************************"
    
    ###########################################################################
    # Names and descriptions
    ###########################################################################
    
    print(boundary, file=stream)
    print("APPROACHES: ", file=stream)   
    print(boundary, file=stream)
    print("", file=stream)
    
    for approach_name in approach_data:
        print("*", approach_name, file=stream)    
        print("\t", A05.get_approach_description(approach_name), file=stream)
        print("", file=stream)  
        
        # Grab at least one model metric list
        model_metrics = approach_data[approach_name]["metrics"]
       
    ###########################################################################   
    # Results
    ###########################################################################
    
    print(boundary, file=stream)
    print("RESULTS:", file=stream)     
    print(boundary, file=stream) 
    
    # Create header
    header = "APPROACH"    
    for data_type in model_metrics:        
        data_metrics = model_metrics[data_type]
        for key in data_metrics:
            header += "\t" + data_type + "_" + key    
    table_data = header + "\n"
    
    # Add data
    for approach_name in approach_data:
        model_metrics = approach_data[approach_name]["metrics"]
        table_data += approach_name
                
        for data_type in model_metrics:        
            data_metrics = model_metrics[data_type]
            for key in data_metrics:
                cell_string = "\t%.4f" % data_metrics[key]
                table_data += cell_string
        table_data += "\n"
        
    print(table_data, file=stream)       
    
    ###########################################################################   
    # Models
    ###########################################################################
                    
    print(boundary, file=stream)
    print("MODEL ARCHITECTURES:", file=stream)       
    print(boundary, file=stream)
    for approach_name in approach_data:
        model = approach_data[approach_name]["model"]
        print("*", approach_name, file=stream)    
        print(model, file=stream)    
        print("", file=stream)          
  
###############################################################################
# MAIN
###############################################################################

def main():     
    # Get names of all approaches
    all_names = A05.get_approach_names()
    
    # Which one?
    print("Approach names (-1 for all):")
    for i in range(len(all_names)):
        print(str(i) + ". " + all_names[i])
    choice = int(input("Enter choice: "))
    
    if choice < 0:
        chosen_approach_names = all_names
    else:
        chosen_approach_names = [all_names[choice]]
    
    approach_data = {}
    
    for approach_name in chosen_approach_names:
        print("EVALUATING APPROACH:", approach_name)    
        approach_data[approach_name] = {}
        
        # Create only the testing data transform    
        transform = A05.get_data_transform(approach_name, training=False)
            
        # Load CIFAR10 data
        training_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
        
        # Create dataloaders
        batch_size = A05.get_batch_size(approach_name)
        train_dataloader = DataLoader(training_data, batch_size=batch_size)
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
            
        # Load up previous weights
        model_path = os.path.join(out_dir, "model_" + approach_name + ".pth")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from:", model_path)
        approach_data[approach_name]["model"] = model
    
        # Evaluate    
        train_eval_data = get_predictions_and_ground(model, train_dataloader, device)
        print("Data acquired from training...")
        test_eval_data = get_predictions_and_ground(model, test_dataloader, device)
        print("Data acquired from testing...")
        
        # Get metric values
        model_metrics = {}
        model_metrics["TRAINING"] = compute_metrics(**train_eval_data)
        model_metrics["TESTING"] = compute_metrics(**test_eval_data)       
        
        # Store model metrics
        approach_data[approach_name]["metrics"] = model_metrics
             
    # Print and save metrics
    print_results(approach_data)
    if len(chosen_approach_names) == 1:
        result_filename = chosen_approach_names[0] + "_RESULTS.txt"
    else:
        result_filename = "ALL_RESULTS.txt"
        
    with open(out_dir + "/" + result_filename, "w") as f:
        print_results(approach_data, stream=f)
    
if __name__ == "__main__": 
    main()
    