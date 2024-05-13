import torch
import torch.nn as nn
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from pytorchtools import EarlyStopping
import models as m

import joblib
import numpy as np
import time

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image


def make_loaders(batch_size, dataset):
    torch.manual_seed(42)

    current_path = os.path.dirname(os.path.realpath(__file__))
    data = os.path.join(current_path, "Data/")

    valid_size = 0.2 #The percentage of training set to use as validation
    
    if dataset == "MNIST":
    
        transform = transforms.ToTensor()

        train_data = datasets.MNIST(root=data,
                                    train=True, 
                                    download=True, 
                                    transform=transform)
        
        test_data = datasets.MNIST(root=data, 
                                   train=False, 
                                   download=True, 
                                   transform=transform)
    elif dataset == "FashionMNIST":
        transform = transforms.ToTensor()

        train_data = datasets.FashionMNIST(root=data,
                                           train=True, 
                                           download=True, 
                                           transform=transform)
        test_data = datasets.FashionMNIST(root=data, 
                                          train=False, 
                                          download=True, 
                                          transform=transform)
    elif dataset == "CATDOG":
        train_transform = transforms.Compose([
        transforms.RandomRotation(10),      # rotate +/- 10 degrees
        transforms.RandomHorizontalFlip(),  # reverse 50% of images
        transforms.Resize(224),             # resize shortest side to 224 pixels
        transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),             # resize shortest side to 224 pixels
            transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])

        train_data = datasets.ImageFolder(os.path.join(data, 'CATS_DOGS/train'), transform=train_transform)
        test_data = datasets.ImageFolder(os.path.join(data, 'CATS_DOGS/test'), transform=test_transform)
    
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
    
    class_names = train_data.classes

    return train_loader, valid_loader, test_loader, class_names

def set_data_params(data):
    if data == 1:
        dataset = "MNIST"
        classes = 10
        in_channels = 1
    elif data == 2:
        dataset = "FashionMNIST"
        classes = 10
        in_channels = 1
    elif data == 3:
        dataset = "CATDOG"
        classes = 2
        in_channels = 3
    else:
        raise Exception(f"The dataset number should be 1, 2 or 3. You gave ({data=})" )

    return dataset, classes, in_channels

def count_parameters(model):
    params = [p.numel() for p in model.parameters()]
    print(f"The model has {sum(params)} trainable parameters\n")

def train_model(model, train_loader, valid_loader, patience, n_epochs):
    if (patience != 3 and patience !=5 and patience != 7):
        raise Exception(f"The patience number should be 3, 5 or 7. You gave ({patience=})" )
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
            
            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device) #Copy data to training device
                model.to(device) #Copy the model onto the training device

            output = model(data) #Forward pass (Calculate predicted outputs)
            loss = criterion(output, target) #Calculate the loss
            loss.backward() #Backward pass (Calculate the gradient of the loss)
            optimizer.step() #Single optimization step (Update parameters)
            train_losses.append(loss.item()) #Store training loss
        
        model.eval() #Prepare the model for evaluation
        for data, target in valid_loader:

            if torch.cuda.is_available():
                data, target = data.to(device), target.to(device) #Copy data to training device
                model.to(device) #Copy the model onto the training device

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

    return model, train_losses, valid_losses

def evaluate_model(model, test_loader, batch_size, classes):
    torch.manual_seed(42)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print("Evaluating on:", device)
    start_time = time.time()

    test_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in test_loader:

        if torch.cuda.is_available():
            data, target = data.to(device), target.to(device) #Copy data to training device
            model.to(device) #Copy the model onto the training device

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

    for i in range(classes):
        if class_total[i] > 0:
            print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
                str(i), 100 * class_correct[i] / class_total[i],
                np.sum(class_correct[i]), np.sum(class_total[i])))
        else:
            print("Test Accuracy of %5s: N/A (no training examples)")

    print("\nTest Accuracy (Overall): %2d%% (%2d/%2d)" % (
        100. * np.sum(class_correct) / np.sum(class_total),
        np.sum(class_correct), np.sum(class_total)))
    
    print(f"\nEvaluation completed in {time.time()-start_time} seconds\n")

def save_model(model, model_name, patience, kernel):
    if (patience != 3 and patience !=5 and patience != 7):
        raise Exception(f"The patience number should be 3, 5 or 7. You gave ({patience=})" )
    if (kernel != 3 and kernel != 5 and kernel != 7):
        raise Exception(f"The kernel number should be 3, 5 or 7. You gave ({kernel=})" )
    
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, "TestModels/")
    patience = "Patience" + str(patience) +"/"
    patience_folder = os.path.join(current_path, patience)
    kernel = "Kernel" + str(kernel) + "/"
    kernel_folder = os.path.join(patience_folder, kernel)
    save_model = os.path.join(kernel_folder, model_name)
    joblib.dump(model, save_model)
    print(f"Model saved as: {model_name}\n")

def load_model(model_name, patience, kernel):
    if (patience != 3 and patience !=5 and patience != 7):
        raise Exception(f"The patience number should be 3, 5 or 7. You gave ({patience=})" )
    if (kernel != 3 and kernel != 5 and kernel != 7):
        raise Exception(f"The kernel number should be 3, 5 or 7. You gave ({kernel=})" )
    
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, "TestModels/")
    patience = "Patience" + str(patience) +"/"
    patience_folder = os.path.join(current_path, patience)
    kernel = "Kernel" + str(kernel) + "/"
    kernel_folder = os.path.join(patience_folder, kernel)
    save_model = os.path.join(kernel_folder, model_name)
    model = joblib.load(save_model, mmap_mode='c')
    print(f"{model_name} has been loaded!\n")
    return model
    
def set_model_name(model, dataset):
    if model == 1:
        filename = dataset + "_CNN_small"
    elif model == 2:
        filename = dataset + "_CNN_medium"
    elif model == 3:
        filename = dataset + "_CNN_big"
    elif model == 4:
        filename = dataset + "_FNN_small"
    elif model == 5:
        filename = dataset + "_FNN_medium"
    elif model == 6:
        filename = dataset + "_FNN_big"
    else:
        raise Exception(f"The model number ({model}) shoul be from 1 to 6. You gave ({model=})")

    savedmodelname = filename + ".pkl"

    return savedmodelname, filename

def set_model(modelnum, kernel, dataset, classes, in_channels):
    if (kernel != 3 and kernel != 5 and kernel != 7):
        raise Exception(f"The kernel number should be 3, 5 or 7. You gave ({kernel=})" )
    if (dataset != 1 and dataset != 2 and dataset != 3):
        raise Exception(f"The dataset number should be 1, 2 or 3. You gave ({dataset=})" )
    if (modelnum == 3 or modelnum == 6) and (dataset == 1 or dataset == 2):
        raise Exception(f"Training a big model ({modelnum}) on this dataset ({dataset}) cannot be done, please choose a smaller model!")

    if modelnum == 1:
        model = m.Conv_NN_small(kernelsize=kernel, 
                                classes=classes,  
                                in_channels=in_channels)
    elif modelnum == 2:
        model = m.Conv_NN_medium(kernelsize=kernel, 
                                 classes=classes, 
                                 in_channels=in_channels)
    elif modelnum == 3:
        model = m.Conv_NN_big(kernelsize=kernel, 
                              classes=classes, 
                              in_channels=in_channels)
    elif modelnum == 4:
        model = m.FNO_NN_small(kernelsize=kernel, 
                               classes=classes, 
                               in_channels=in_channels)
    elif modelnum == 5:
        model = m.FNO_NN_medium(kernelsize=kernel, 
                                classes=classes, 
                                in_channels=in_channels)
    elif modelnum == 6:
        model = m.FNO_NN_big(kernelsize=kernel, 
                             classes=classes, 
                             in_channels=in_channels)
    else:
        raise Exception(f"The model number ({modelnum}) shoul be from 1 to 6. You gave ({modelnum=})")
        
    return model

def example_plot(train_loader, dataset):
    np.set_printoptions(formatter=dict(int=lambda x: f"{x:4}"))

    for images, labels in train_loader:
        break
    
    if dataset == "CATDOG":
        im = make_grid(images, nrow=5)

        inv_normalize = transforms.Normalize( mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                          std=[1/0.229, 1/0.224, 1/0.225])
        test_transform = transforms.Compose([
                         transforms.Resize(224),             # resize shortest side to 224 pixels
                         transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ])

        current_path = os.path.dirname(os.path.realpath(__file__))
        data = os.path.join(current_path, "Data/CATS_DOGS")
        test_data = datasets.ImageFolder(os.path.join(data, 'test'), transform=test_transform)

        rand_indices = np.random.randint(low=1,high=6000,size=10)
        fig = plt.figure(figsize=(25,4))
        for rand_idx, idx in zip(rand_indices, np.arange(10)):
            ax = fig.add_subplot(1, 10, idx+1, xticks=[], yticks=[])
            im = inv_normalize(test_data[rand_idx][0])
            ax.imshow(np.transpose(im.numpy(),(1, 2, 0)))
        plt.show()
    else:
        images = images.numpy()

        fig = plt.figure(figsize=(25,4))
        for idx in np.arange(10):
            ax = fig.add_subplot(1, 10, idx+1, xticks=[], yticks=[])
            ax.imshow(np.squeeze(images[idx]), cmap="gray")
        plt.show()
    
    figname = dataset + "_example_plot.png"
    fig.savefig(figname, bbox_inches="tight")

def loss_plot(train_loss, valid_loss, filename, patience, kernel):
    if (patience != 3 and patience !=5 and patience != 7):
        raise Exception(f"The patience number should be 3, 5 or 7. You gave ({patience=})" )
    if (kernel != 3 and kernel != 5 and kernel != 7):
        raise Exception(f"The kernel number should be 3, 5 or 7. You gave ({kernel=})" )
    
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

def sample_test(model, test_loader, filename, patience, kernel, dataset):
    if (patience != 3 and patience !=5 and patience != 7):
        raise Exception(f"The patience number should be 3, 5 or 7. You gave ({patience=})" )
    if (kernel != 3 and kernel != 5 and kernel != 7):
        raise Exception(f"The kernel number should be 3, 5 or 7. You gave ({kernel=})" )
    
    for images, labels in test_loader:
        break

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        images = images.to(device)
        model.to(device)

    output = model(images)
    _, preds = torch.max(output, 1)

    images = images.detach().cpu().numpy()

    if dataset == "CATDOG":
        inv_normalize = transforms.Normalize( mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                                          std=[1/0.229, 1/0.224, 1/0.225])
    
        test_transform = transforms.Compose([
                         transforms.Resize(224),             # resize shortest side to 224 pixels
                         transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
                         transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406],
                                              [0.229, 0.224, 0.225])
        ])

        current_path = os.path.dirname(os.path.realpath(__file__))
        data = os.path.join(current_path, "Data/CATS_DOGS")
        test_data = datasets.ImageFolder(os.path.join(data, 'test'), transform=test_transform)

        fig = plt.figure(figsize=(25,4))
        for idx in np.arange(20):
            ax = fig.add_subplot(2, 10, idx+1, xticks=[], yticks=[])
            im = inv_normalize(test_data[idx][0])
            ax.imshow(np.transpose(im.numpy(),(1, 2, 0)))
            ax.set_title("{} ({})".format(str(preds[idx].item()),
                                          str(labels[idx].item())),
                                          color=("g" if preds[idx]==labels[idx] else "r"))
        plt.show()
    else:
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

def make_confusion_matrix(model, test_loader, class_names, filename, patience, kernel, dataset):
    if (patience != 3 and patience !=5 and patience != 7):
        raise Exception(f"The patience number should be 3, 5 or 7. You gave ({patience=})" )
    if (kernel != 3 and kernel != 5 and kernel != 7):
        raise Exception(f"The kernel number should be 3, 5 or 7. You gave ({kernel=})" )
    
    test_preds = []
    class_labels = []

    for data in test_loader:
        model.cpu() #Copy the model back to cpu
        inputs, labels = data #Unpack the data into inputs and labels
        output = model(inputs) #Forward pass (Calculate predicted outs)
        _, pred = torch.max(output, 1) #Convert output probabilities to pred
        test_preds.extend(pred.numpy()) #Store predictions
        class_labels.extend(labels.numpy()) #Store the class labels

    #Convert lists into numpy arrays
    test_preds = np.array(test_preds)
    class_labels = np.array(class_labels)
    
    conf_matrix = confusion_matrix(class_labels,test_preds)
    
    # Plot the confusion matrix.
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    current_path = os.path.dirname(os.path.realpath(__file__))
    current_path = os.path.join(current_path, "TestPlots/")
    patience = "Patience" + str(patience) +"/"
    patience_folder = os.path.join(current_path, patience)
    kernel = "Kernel" + str(kernel) + "/"
    kernel_folder = os.path.join(patience_folder, kernel)
    figname = filename + "_confusion_matrix_plot.png"
    save_plot = os.path.join(kernel_folder, figname)
    fig.savefig(save_plot, bbox_inches="tight")

def user_test(img, class_names):
    CNN_model  = load_model("CATDOG_CNN_big.pkl",7,7)
    CNN_model = CNN_model.to("cpu")
    FNN_model  = load_model("CATDOG_FNN_big.pkl",7,7)
    FNN_model = FNN_model.to("cpu")

    current_path = os.path.dirname(os.path.realpath(__file__))
    img_path = os.path.join(current_path, "UserTest/", img)
    img = Image.open(img_path)
    
    test_transform = transforms.Compose([
            transforms.Resize(224),             # resize shortest side to 224 pixels
            transforms.CenterCrop(224),         # crop longest side to 224 pixels at center
            transforms.ToTensor()
                                ])
    img = test_transform(img)
    class_names = np.array(class_names)
    preds = []
    classnameidx = []
    preds = np.array(preds)
    classnameidx = np.array(classnameidx, dtype=int)

    CNN_model.eval()
    with torch.no_grad():
        new_pred_CNN = CNN_model(img.view(1,3,224,224)).argmax()
    if new_pred_CNN.item() == 0:
        pred_CNN = "CAT"
        preds = np.append(preds, pred_CNN)
        classnameidx = np.append(classnameidx, 0)
    else: #new_pred_CNN.item() == 1
        pred_CNN = "DOG"
        preds = np.append(preds, pred_CNN)
        classnameidx = np.append(classnameidx, 1)
    print(f'CNN:\n Predicted value: {pred_CNN}\n True Value: {class_names[new_pred_CNN.item()]}')
    

    FNN_model.eval()
    with torch.no_grad():
        new_pred_FNN = FNN_model(img.view(1,3,224,224)).argmax()
    if new_pred_FNN.item() == 0:
        pred_FNN = "CAT"
        preds = np.append(preds, pred_CNN)
        classnameidx = np.append(classnameidx, 0)
    else: #new_pred_FNN.item() == 1
        pred_FNN = "DOG"
        preds = np.append(preds, pred_CNN)
        classnameidx = np.append(classnameidx, 1)
    print(f'FNN:\n Predicted value: {pred_FNN}\n True Value: {class_names[new_pred_FNN.item()]}')
    preds = np.append(preds, pred_FNN)
    classnameidx = np.append(classnameidx, new_pred_FNN.item())

    fig = plt.figure(figsize=(10,4))
    for idx in np.arange(2):
        ax = fig.add_subplot(1, 2, idx+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(img.numpy(),(1, 2, 0)))
        ax.set_title("{} ({})".format(str(preds[idx].item()),
                                      str(class_names[classnameidx[idx].item()].item())),
                                      color=("g" if preds[idx].item()==class_names[classnameidx[idx].item()].item() else "r"))
    plt.show()

    current_path = os.path.dirname(os.path.realpath(__file__))
    figname = "ImageTest.png"
    save_plot = os.path.join(current_path, figname)
    fig.savefig(save_plot, bbox_inches="tight")