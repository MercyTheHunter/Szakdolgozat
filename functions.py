import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pytorchtools import EarlyStopping

import joblib
import numpy as np
import time

import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix as mcm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
#from yellowbrick.classifier import ConfusionMatrix
#from yellowbrick.classifier import ClassPredictionError
#from yellowbrick.classifier import ROCAUC

def MNIST_make_loaders(batch_size):
    torch.manual_seed(42)

    current_path = os.path.dirname(os.path.realpath(__file__))
    data = os.path.join(current_path, "Data/")
    valid_size = 0.2 #The percentage of training set to use as validation
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(root=data,
                                train=True, 
                                download=True, 
                                transform=transform)
    test_data = datasets.MNIST(root=data, 
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

def FashionMNIST_make_loaders(batch_size):
    torch.manual_seed(42)

    valid_size = 0.2 #The percentage of training set to use as validation
    transform = transforms.ToTensor()

    train_data = datasets.FashionMNIST(root='../Data',
                                train=True, 
                                download=True, 
                                transform=transform)
    test_data = datasets.FashionMNIST(root='../Data', 
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

def STL10_make_loaders(batch_size):
    torch.manual_seed(42)

    valid_size = 0.2 #The percentage of training set to use as validation
    transform = transforms.ToTensor()

    train_data = datasets.STL10(root='../Data',
                                split="train", 
                                download=True, 
                                transform=transform)
    test_data = datasets.STL10(root='../Data', 
                               split="test", 
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

def count_parameters(model):
    params = [p.numel() for p in model.parameters()]
    print(f"The model has {sum(params)} trainable parameters\n")

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
        print_msg = (f"[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] " +
                     f"train_loss: {train_loss:.5f} " +
                     f"valid_loss: {valid_loss:.5f} ")
        print(print_msg)

        #Clear lists for the next epoch
        train_losses = []
        valid_losses = []

        #Early_stopping checks if the validation loss has decreased, 
        #if it has, then it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping\n")
            break

    print(f"Training completed in {time.time()-start_time} seconds\n")

    #Load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses

def evaluate_model(model, test_loader, batch_size):
    torch.manual_seed(42)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Evaluating on:", device)
    start_time = time.time()

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:
        if len(target.data) != batch_size:
            break
        output = model(data) #Forward pass (Calculate predicted outs)
        loss = criterion(output, target) #Calculate the loss
        test_loss += loss.item()*data.size(0) #Update test loss
        _, pred = torch.max(output, 1) #Convert output probabilities to pred
        correct = np.squeeze(pred.eq(target.data.view_as(pred))) #Compare pred to the truth
        for i in range(batch_size): #Calculate test accuracy for each class
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1
    #Calculate and print the average test loss
    test_loss = test_loss / len(test_loader.dataset)
    print("Test Loss: {:.6f}\n".format(test_loss))

    for i in range(10):
        if class_total[i] > 0:
            print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print("Test Accuracy of %5s: N/A (no training examples)")

    print("\nTest Accuracy (Overall): %2d%% (%2d/%2d)" % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))

def save_model(model, model_name, patience, kernel):
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, "TestModels/")
    patience = "Patience" + str(patience) +"/"
    patience_folder = os.path.join(current_path, patience)
    kernel = "Kernel" + str(kernel) + "/"
    kernel_folder = os.path.join(patience_folder, kernel)
    save_model = os.path.join(kernel_folder, model_name)
    joblib.dump(model, save_model)
    print(f"Model saved as: {model_name}\n")

def load_model(model_name):
    current_path = os.path.dirname(os.path.realpath(__file__))
    saved_model_name = os.path.join(current_path, "SavedModels/", model_name)
    model = joblib.load(saved_model_name)
    print(f"{model_name} has been loaded!\n")
    return model
    
def example_plot(train_loader, dataset):
    np.set_printoptions(formatter=dict(int=lambda x: f"{x:4}"))

    for images, labels in train_loader:
        break

    images = images.numpy()

    fig = plt.figure(figsize=(25,4))
    for idx in np.arange(10):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap="gray")
    plt.show()
    figname = dataset + "_example_plot.png"
    fig.savefig(figname, bbox_inches="tight")

def loss_plot(train_loss, valid_loss, filename, patience, kernel):
    #Loss during the training process
    fig = plt.figure(figsize=(10,8))
    plt.plot(range(1, len(train_loss)+1), train_loss, label="Training Loss")
    plt.plot(range(1, len(valid_loss)+1), valid_loss, label="Validation Loss")

    #Lowest validation loss
    minposs = valid_loss.index(min(valid_loss))+1
    plt.axvline(minposs, linestyle="--", color="r", label="Early Stopping Checkpoint")

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.ylim(0, 3)
    plt.xlim(0, len(train_loss)+1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, "TestPlots/")
    patience = "Patience" + str(patience) +"/"
    patience_folder = os.path.join(current_path, patience)
    kernel = "Kernel" + str(kernel) + "/"
    kernel_folder = os.path.join(patience_folder, kernel)
    figname = filename + "_loss_plot.png"
    save_plot = os.path.join(kernel_folder, figname)
    fig.savefig(save_plot, bbox_inches="tight")

def sample_test(model, test_loader, filename, patience, kernel):
    for images, labels in test_loader:
        break

    output = model(images)
    _, preds = torch.max(output, 1)
    images = images.numpy()

    fig = plt.figure(figsize=(25,4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap="gray")
        ax.set_title("{} ({})".format(str(preds[idx].item()),
                                      str(labels[idx].item())),
                                      color=("g" if preds[idx]==labels[idx] else "r"))
    plt.show()
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, "TestPlots/")
    patience = "Patience" + str(patience) +"/"
    patience_folder = os.path.join(current_path, patience)
    kernel = "Kernel" + str(kernel) + "/"
    kernel_folder = os.path.join(patience_folder, kernel)
    figname = filename + "_sample_test_plot.png"
    save_plot = os.path.join(kernel_folder, figname)
    fig.savefig(save_plot, bbox_inches="tight")

    
