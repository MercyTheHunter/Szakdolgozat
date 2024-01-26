import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
import time

def make_loaders():
    torch.manual_seed(42)

    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(root='../Data', train=True, download=True, transform=transform)
    test_data = datasets.FashionMNIST(root='../Data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
    test_loader = DataLoader(test_data,batch_size=10,shuffle=False)
    
    return train_loader, test_loader

def example_plot(train_loader):
    np.set_printoptions(formatter=dict(int=lambda x: f"{x:4}"))

    for images in train_loader:
        break

    im = make_grid(images[:10],nrow=10)
    plt.figure(figsize=(10,4))
    plt.imshow(np.transpose(im.numpy(),(1,2,0)))

def count_parameters(model):
    params = [p.numel() for p in model.parameters()]
    print(sum(params))

def train_model(model, train_loader):
    torch.manual_seed(42)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Training on:", device)
    start_time = time.time()

    epochs = 50
    train_losses = []
    train_correct = []

    for i in range(epochs):
        train_acc = 0.0
        train_num = 0
        train_loss = 0.0

        b = 0
        for X_train, y_train in train_loader:
            b+=1

            X_train = X_train.to(device)
            y_train = y_train.to(device)

            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            train_loss += loss.item()

            # which class
            predicted = torch.max(y_pred.data,1)[1]
            train_acc += (predicted == y_train).sum()
            train_num += 10 #BatchSize

            # gradient calculating ---> update the weights
            optimizer.zero_grad()
            loss.backward() # gradient
            optimizer.step()

            # print results
            if b%1000 == 0:
                print(f"[{i+1}/{epochs}] batch: [{b}/{len(train_loader)}] loss: {loss.item():.3f} acc: {train_acc.item()/train_num:.3f}")


        train_losses.append(train_loss)
        train_correct.append(train_acc.item()/train_num)
    print(f"Training completed in {time.time()-start_time} seconds")

def evaluate_model(model, test_loader):
    torch.manual_seed(42)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Evaluating on:", device)
    start_time = time.time()

    test_acc = 0.0
    test_num = 0
    test_loss = 0.0
    with torch.no_grad():
        for X_test, y_test in test_loader:

            X_test = X_test.to(device)
            y_test = y_test.to(device)
            y_pred = model(X_test)

            #Evaluating accuracy
            predicted = torch.max(y_pred.data,1)[1]
            test_acc += (predicted == y_test).sum()
            test_num += 10 #BatchSize

            #Evaluating loss
            loss = criterion(y_pred,y_test)
            test_loss += loss.item()

        print(f"\n Test loss: {test_loss:.3f}\n Test accuracy: {test_acc.to(int)}/{test_num} = {(test_acc/test_num)*100:.2f}% \n")
    print(f"Evaluation completed in {time.time()-start_time} seconds")