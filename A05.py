import torch
from torch import nn 
from torchvision import models
from torchvision.transforms import v2
import math


class SimpleNetwork(nn.Module):
    # Taken from: TorchLand.py, which was discussed in the class
    """
    APPROACH        TRAINING_accuracy       TRAINING_f1     TESTING_accuracy        TESTING_f1
    simple_network  0.9512                  0.9512          0.7185                  0.7179
    """
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

class VanillaCNN(nn.Module):
    """
    APPROACH        TRAINING_accuracy       TRAINING_f1     TESTING_accuracy        TESTING_f1
    vanilla_cnn     0.6851                  0.6814          0.6431                  0.6393
    """
    def __init__(self, class_cnt):
        super().__init__()        
        self.net_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            
            nn.Flatten(),
            
            nn.Linear(1600, 32),
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )

    def forward(self, x):        
        logits = self.net_stack(x)
        return logits

class VanillaCNNWithDropOut(nn.Module):
    """
    APPROACH                        TRAINING_accuracy       TRAINING_f1     TESTING_accuracy        TESTING_f1
    vanilla_cnn_with_dropout        0.6727                  0.6677          0.6496                  0.6443
    """
    def __init__(self, class_cnt):
        super().__init__()   
        
        self.stack = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
            nn.Dropout(0.25),
            
            nn.Flatten(),
            
            nn.Linear(1600, 32),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(32, class_cnt)
        )

    def forward(self, x):        
        logits = self.stack(x)
        return logits

class DeepCNN(nn.Module):
    """
    APPROACH        TRAINING_accuracy       TRAINING_f1     TESTING_accuracy        TESTING_f1
    deep_cnn        0.9198                  0.9203          0.7903                  0.7919
    """
    def __init__(self, class_cnt):
        super().__init__()
        
        self.stack = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Flatten(),
            nn.Linear(128 * (32 // 8) * (32 // 8), 32),
            nn.ReLU(),
            nn.Linear(32, class_cnt)
        )
    
    def forward(self, x):        
        logits = self.stack(x)
        return logits

class DeepCNNWithBatchNormDropOut(nn.Module):
    """
    APPROACH                            TRAINING_accuracy       TRAINING_f1     TESTING_accuracy        TESTING_f1
    deep_cnn_with_batchnorm_dropout     0.8359                  0.8354          0.8036                  0.8027
    """
    def __init__(self, class_cnt):
        super().__init__()
        
        self.stack = nn.Sequential(
            
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=1, padding='same'),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(128 * (32 // 8) * (32 // 8), 32),
            nn.Dropout(0.25),
            nn.Linear(32, class_cnt)
        )
    
    def forward(self, x):        
        logits = self.stack(x)
        return logits

class VGG19Bn(nn.Module):
    """
    APPROACH        TRAINING_accuracy       TRAINING_f1     TESTING_accuracy        TESTING_f1
    vgg19_bn        0.9981                  0.9981          0.9131                  0.9127
    """
    # Taken from: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
    def __init__(self, features, class_cnt):
        super(VGG19Bn, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, class_cnt),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class ResNet34(nn.Module):
    """
    APPROACH        TRAINING_accuracy       TRAINING_f1     TESTING_accuracy        TESTING_f1
    resnet_34       0.9974                  0.9974          0.9641                  0.9641
    """
    
    # Modified the classification layer from VGG19Bn and desinged this classifier.
    def __init__(self, features, class_cnt):
        super(ResNet34, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.LeakyReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.LeakyReLU(True),
            nn.Linear(128, class_cnt)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class EfficientNetB7():
    
    def __init__(self, features, class_cnt):
        super(EfficientNetB7, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Linear(128, class_cnt)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def get_approach_names():
    approaches = ["simple_network", "vanilla_cnn", "vanilla_cnn_with_dropout", "deep_cnn", "deep_cnn_with_batchnorm_dropout", "vgg19_bn", "resnet_34", "efficient_net_b7"]
    return approaches

def get_approach_description(approach_name):
    description = ""
    if approach_name == "simple_network":
        description = ""
    elif approach_name == "vanilla_cnn":
        description = "It's a plain CNN model that has two convolutional layers."
    elif approach_name == "vanilla_cnn_with_dropout":
        description = "It's a plain CNN model that has two convolutional layers with a dropout layer to avoid overfitting for each conv layer and for linear layer."
    elif approach_name == "deep_cnn":
        description = "As the name goes, it has 3 convolutional blocks to learn more complex features."
    elif approach_name == "deep_cnn_with_batchnorm_dropout":
        description = "3 convolutional blocks with additional added layers like batch_norm and dropout to generalise the model and avoid overfitting"
    elif approach_name == "vgg19_bn":
        description = "Fine tuning pre-trained VGG19 model with batch normalization."
    elif approach_name == "resnet_34":
        description = "Fine tuning pre-trained ResNet 34 model by adding a custom classification layer with leakyrelu as activation function."
    elif approach_name == "efficient_net_b7":
        description = "Fine tuning pre-trained EfficientNet b0 model by adding a custom classification layer with leakyrelu as activation function."
    return description

def get_data_transform(approach_name, training):
    data_transform = v2.Compose([v2.ToTensor(), v2.ConvertImageDtype()])
    
    if approach_name in ["deep_cnn", "deep_cnn_with_batchnorm_dropout"]:
        if training is True:
            data_transform.transforms.insert(1, v2.RandomHorizontalFlip())
    
    elif approach_name in ['vgg19_bn']:
        # Taken from: https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/main.py
        if training is True:
            data_transform.transforms.insert(1, v2.RandomHorizontalFlip())
            data_transform.transforms.insert(2, v2.RandomCrop(32, 4))
            data_transform.transforms.insert(3, v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        if training is False:
            data_transform.transforms.insert(1, v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
    
    elif approach_name in ['resnet_34']:
        # Taken from: https://www.kaggle.com/code/francescolorenzo/96-fine-tuning-resnet34-with-pytorch
        if training is True:
            data_transform.transforms.insert(1, v2.Resize((224, 224)))
            data_transform.transforms.insert(2, v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10))
            data_transform.transforms.insert(3, v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        
        if training is False:
            data_transform.transforms.insert(1, v2.Resize((224, 224)))
            data_transform.transforms.insert(2, v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
            
    elif approach_name in ['efficient_net_b7']:
        if training is True:
            data_transform.transforms.insert(1, v2.Resize((224, 224)))
            # data_transform.transforms.insert(2, v2.AutoAugment(policy=v2.AutoAugmentPolicy.CIFAR10))
            data_transform.transforms.insert(3, v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        
        if training is False:
            data_transform.transforms.insert(1, v2.Resize((224, 224)))
            data_transform.transforms.insert(2, v2.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))

    return data_transform
    
def get_batch_size(approach_name):
    if approach_name == "simple_network":
      return 64  
    elif approach_name == "vanilla_cnn":
        return 32
    elif approach_name == "vanilla_cnn_with_dropout":
        return 64
    elif approach_name == "deep_cnn":
        return 64
    elif approach_name == "deep_cnn_with_batchnorm_dropout":
        return 64
    elif approach_name in ["vgg19_bn", "resnet_34", 'efficient_net_b7']:
        return 128
    
def create_model(approach_name, class_cnt):
    if approach_name == "simple_network":
        simple_network = SimpleNetwork(class_cnt=class_cnt)  
        return simple_network
    
    elif approach_name == "vanilla_cnn":
        vanilla_cnn = VanillaCNN(class_cnt=class_cnt)
        return vanilla_cnn
    
    elif approach_name == "vanilla_cnn_with_dropout":
        vanilla_cnn_with_dropout = VanillaCNNWithDropOut(class_cnt=class_cnt)
        return vanilla_cnn_with_dropout
    
    elif approach_name == "deep_cnn":
        deep_cnn = DeepCNN(class_cnt=class_cnt)
        return deep_cnn
    
    elif approach_name == "deep_cnn_with_batchnorm_dropout":
        deep_cnn_with_batchnorm_dropout = DeepCNNWithBatchNormDropOut(class_cnt=class_cnt)
        return deep_cnn_with_batchnorm_dropout

    elif approach_name == "vgg19_bn":
        # Taken from: https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html
        
        def make_layers(cfg, batch_norm=False):
            layers = []
            in_channels = 3
            for v in cfg:
                if v == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = v
            return nn.Sequential(*layers)
        
        cfg = {
            'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
        }
        
        vgg = VGG19Bn(make_layers(cfg['E'], batch_norm=True), class_cnt)    
        
        return vgg
    
    elif approach_name == "resnet_34":
        
        resnet = models.resnet34(pretrained=True)        
        
        # Taken from : https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/63?page=4
        features = nn.Sequential(*list(resnet.children())[:-1])
        
        resnet_34 = ResNet34(features, class_cnt)
        return resnet_34
    
    
    elif approach_name == "efficient_net_b7":
        
        efficientnet = models.efficientnet_b0(pretrained=True)        
        # efficientnet[1] = nn.Linear(efficientnet.in_features, class_cnt)
        # efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, class_cnt)
        efficient_net = EfficientNetB7(efficientnet.classifier[1].in_features, class_cnt)

        return efficient_net
        
        efficient_net_b7 = EfficientNetB7().efficient_net_b7(class_cnt)
        
        return efficient_net_b7
    
def train_model(approach_name, model, device, train_dataloader, test_dataloader):
    size = len(train_dataloader.dataset)
    loss_func = nn.CrossEntropyLoss()
    
    if approach_name == "simple_network":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 25
        
    if approach_name in ["vanilla_cnn", "vanilla_cnn_with_dropout"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 25

    elif approach_name in ["deep_cnn", "deep_cnn_with_batchnorm_dropout"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 50   
    
    elif approach_name == "vgg19_bn":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 20#300
    
    elif approach_name in ["resnet_34"]:
        # params_1x = [param for name, param in model.named_parameters() if 'fc' not in str(name)]
        lr, weight_decay = 1e-5, 5e-4
        # optimizer = torch.optim.Adam([{'params':params_1x}, {'params': model.classifier.parameters(), 'lr': lr*10}], lr=lr, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        epochs = 10
        
    elif approach_name in ["efficient_net_b7"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 5
    
    for each_epoch in range(epochs):
        print("##################### epoch", each_epoch, "#########################" )
        model.train()
        
        for batch, (X,y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            
            pred = model(X)
            loss = loss_func(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if batch%100 == 0:
                loss = loss.item()
                index = (batch+1)*len(X)
                print(index, "of", size, ": Loss =", loss)
        
    return model
