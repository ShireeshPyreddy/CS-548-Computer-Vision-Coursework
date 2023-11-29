import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def main():
    data_transform = v2.Compose([
        v2.ToImageTensor(),
        v2.ConvertDtype()
    ])
    
    train_data = datasets.CIFAR10(train=True, 
                                  root="data",
                                  download=True, 
                                  transform=data_transform)
    
    test_data = datasets.CIFAR10(train=False,
                                 root="data",
                                 download=True,
                                 transform=data_transform)
    
    batch_size = 64
    train_dataloader = DataLoader(train_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_data,
                                 batch_size=batch_size)
    
    
    train_iter = iter(train_dataloader)    
    for _ in range(5):
        X,y = next(train_iter)
        print(X.shape, y.shape)
        
        X = X.numpy()
        X = X[0]
        X = np.transpose(X, [1,2,0])
        X = cv2.cvtColor(X, cv2.COLOR_RGB2BGR)
        X = cv2.resize(X, dsize=None, fx=5.0, fy=5.0)
        
        cv2.imshow("IMAGE", X)
        cv2.waitKey(-1)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    