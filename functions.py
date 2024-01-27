import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pytorchtools import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt
import time

def make_loaders(batch_size):
    torch.manual_seed(42)

    valid_size = 0.2 #The percentage of training set to use as validation
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='../Data',
                                train=True, 
                                download=True, 
                                transform=transform)
    test_data = datasets.MNIST(root='../Data', 
                               train=False, 
                               download=True, 
                               transform=transform)
    #Obtain the indices from the training set that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    #Define the samplers to obtaiun the training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)


    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              sampler=train_sampler)
    valid_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              sampler=valid_sampler)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader

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

def train_model(model, train_loader, valid_loader, patience, n_epochs):
    torch.manual_seed(42)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Training on:", device)
    start_time = time.time()

    train_losses = []
    valid_losses = []
    
    avg_train_losses = []
    avg_valid_losses = []

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        model.train() #Prepare the model for training
        for batch, (data, target) in enumerate(train_loader, 1):

            optimizer.zero_grad() #Clear the gradients (Clear optimized parameters)
            output = model(data) #Forward pass (Calculate predicted outputs)
            loss = criterion(output, target) #Calculate the loss
            loss.backward() #Backward pass (Calculate the gradient of the loss)
            optimizer.step() #Single optimization step (Update parameters)
            train_losses.append(loss.item()) #Store training loss
        
        model.eval() #Prepare the model for evaluation
        for data, target in valid_loader:
            output = model(data) #Forward pass (Calculate predicted outputs)
            loss = criterion(output, target) #Calculate the loss
            valid_losses.append(loss.item()) #Store validation loss
            
            
        
        #Calculate the average loss over an epochs
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
                #Print training and validation statistics
        epoch_len = len(str(n_epochs))
        print_msg = (f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}]" +
                     f"train_loss: {train_loss:.5f}" +
                     f"valid_loss: {valid_loss:.5f}")
        print(print_msg)

        #Clear lists for the next epoch
        train_losses = []
        valid_losses = []

        #Early_stopping checks if the validation loss has decreased, 
        #if it has, then it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

    print(f"Training completed in {time.time()-start_time} seconds")

    #Load the last checkpoint with the best model
    model. load_state_dict(torch.load('checpoint.pt'))

    return model, avg_train_losses, avg_valid_losses

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