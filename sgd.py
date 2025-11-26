#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import main

def train_sgd():
    if torch.cuda.is_available():
        device = torch.device("cuda")  # uses GPU (setup CUDA toolkit with compatible version if not already done with PyTorch, or just use CPU)
    else:
        device = torch.device("cpu")  # uses CPU

    print(f'Using device: {device}')

    # initialises CNN and moves tensors to device
    network = main.CNN()
    network.to(device)

    trainloader, testloader = main.get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    # loss function and optimiser (stochastic gradient descent baseline)
    optimiser = optim.SGD(network.parameters(), lr=main.LEARNING_RATE, momentum=main.MOMENTUM)

    # standard training loop that repeats for [NUM_EPOCHS] times with forward pass, loss calculation then backpropogation, updating weights
    print("Starting training\n")
    for epoch in range(main.NUM_EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader):  # loads batch data
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimiser.zero_grad()
            outputs = network(inputs) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimiser.step()
            running_loss += loss.item()
            if i % 200 == 199:
                print("Epoch [%d/%d], Step [%d/%d], Loss: %.4f"
                      % (epoch+1, main.NUM_EPOCHS, i+1, len(trainloader), running_loss/200))
                running_loss = 0.0

    print('Finished training')

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

    accuracy = correct/total
    print('Accuracy of the network on the 10000 test images: %d %%' % (accuracy*100))
    
    print("Freezing all layers and randomising fc3")

    torch.save(network, "sgd.pt")

    for param in network.parameters():
        param.requires_grad = False  # freezes all layers

    for param in network.fc3.parameters():
        param.requires_grad = True  # unfreezes just fc3

    network.fc3.reset_parameters()  # randomises fc3 weights

    torch.save(network, "base.pt")
    
    print("Saved frozen model with unfrozen fc3. Resetted fc3 parameters and stored in 'base.pt.'")

if __name__ == '__main__':
    train_sgd()
