import torch
from torch import nn 
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

class SimpleNetwork(nn.Module):
    def __init__(self, class_cnt):
        super().__init__()        
        self.net_stack = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32,32, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Flatten(),
            
            nn.Linear(4096, 32),
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )
        
    def forward(self, x):        
        logits = self.net_stack(x)
        return logits
    
def test(model, loss_fn, device, dataloader, dataname):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for X,y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
    
    print(dataname, ":")
    print("\tAccuracy:", correct)
    print("\tLoss:", test_loss)
    
def train_one_epoch(model, loss_fn, optimizer, device, dataloader):
    size = len(dataloader.dataset)
    model.train()
    
    for batch, (X,y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        
        pred = model(X)
        loss = loss_fn(pred, y)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch%100 == 0:
            loss = loss.item()
            index = (batch+1)*len(X)
            print(index, "of", size, ": Loss =", loss)
    

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
    
    
    '''
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
    '''
    
    model = SimpleNetwork(class_cnt=10)
    
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    print(f"Using {device} device")
    
    model = model.to(device)    
    print(model)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    total_epochs = 2
    
    for epoch in range(total_epochs):
        print("** EPOCH", (epoch+1), "**")
        train_one_epoch(model, loss_fn, optimizer, device, 
                        train_dataloader)
        test(model, loss_fn, device, train_dataloader, "TRAIN")
        test(model, loss_fn, device, test_dataloader, "TEST")     
        
        
    model_filename = "mymodel.pth"
    torch.save(model.state_dict(), model_filename)
    
    model2 = SimpleNetwork(class_cnt=10).to(device)
    model2.load_state_dict(torch.load(model_filename))
    
    print("RELOADED MODEL:")
    test(model2, loss_fn, device, train_dataloader, "TRAIN")
    test(model2, loss_fn, device, test_dataloader, "TEST")     
    
if __name__ == "__main__":
    main()
    