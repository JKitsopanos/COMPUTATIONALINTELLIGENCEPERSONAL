#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.transforms as transforms, torchvision, matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F

# hyperparameters
learning_rate = 0.001
momentum = 0.9
num_epochs = 1

# in CIFAR10, each image is 32x32 pixels with 3 color channels (red, blue, green))

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CNN model structure 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1) #3 colour channels in, 64 feature maps out
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # downsamples image by factor of 2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) # 64 features in, 128 features out
        self.fc1 = nn.Linear(128 * 8 * 8, 1024) # 8x8 after 2 poolings, 128 channels
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10) # 10 class output
        self.flatten = nn.Flatten() # converts into 1D feature vector

    # flow of data through the network
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

if __name__ == '__main__':
    
    # multiprocessing and batch size
    dataload_speed = 3
    batch_size = 4

    # load CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', 
                                            train=True, 
                                            download=True,
                                            transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              shuffle=True, 
                                              num_workers=dataload_speed)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                          train=False,
                                          download=True,
                                          transform=transform)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=batch_size,
                                             shuffle=False, 
                                             num_workers=dataload_speed)

    classes = {"airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"}

    if torch.cuda.is_available():
        device = torch.device("cuda")  # uses GPU (setup CUDA toolkit with compatible version if not already done with PyTorch, or just use CPU)
    else:
        device = torch.device("cpu")  # uses CPU

    print(f'Using device: {device}')

    # initialises CNN and moves tensors to device
    network = CNN()
    network.to(device)

    # loss function and optimiser (stochastic gradient descent baseline)
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    # standard training loop that repeats for [num_epochs] times with forward pass, loss calculation then backpropogation, updating weights
    print("Starting Training:\n")
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0): # loads batch data
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            outputs = network(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()

            # print results every specified number of batches
            if i % 200 == 199:
                print("Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
                      % (epoch + 1, num_epochs, i + 1, len(trainloader), running_loss / 200))
                running_loss = 0.0

    print('Finished Training')


    # testing loop, applying trained model to test dataset and calculating accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy))
    
    
    print("\nFreezing all layers and randomising fc3:")

    for param in network.parameters():
        param.requires_grad = False # freezes all layers

    for param in network.fc3.parameters():
        param.requires_grad = True # unfreezes just fc3

    network.fc3.reset_parameters() # randomises fc3 weights
