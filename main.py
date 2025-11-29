#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# hyperparameters
LEARNING_RATE = 0.001
MOMENTUM = 0.9
NUM_EPOCHS = 10
DATALOAD_SPEED = 3
BATCH_SIZE = 4

# in CIFAR10, each image is 32×32 pixels with 3 color channels (red, blue, green))
CLASSES = ["airplane",
           "automobile",
           "bird",
           "cat",
           "deer",
           "dog",
           "frog",
           "horse",
           "ship",
           "truck"]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)  # 3 colour channels in, 64 feature maps out
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # downsamples image by factor of 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)  # 64 features in, 128 features out
        self.fc1 = nn.Linear(128*8*8, 1024)  # 8×8 after 2 poolings, 128 channels
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)  # 10 class output
        self.flatten = nn.Flatten()  # converts into 1D feature vector

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def get_dataloaders():
    # load CIFAR10 dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True, 
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=BATCH_SIZE, 
                                              shuffle=True,
                                              num_workers=DATALOAD_SPEED)
    testset = torchvision.datasets.CIFAR10(root='./data',
                                           train=False, 
                                           download=True,
                                           transform=transform)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=BATCH_SIZE, 
                                             shuffle=False,
                                             num_workers=DATALOAD_SPEED)
    return trainloader, testloader
