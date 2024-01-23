import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

class Dset(Dataset):

    train_data
    test_data

    def __init__(self):
        transform = transforms.ToTensor()

        self.train_data = datasets.FashionMNIST(root='../Data', train=True, download=True, transform=transform)
        self.test_data = datasets.FashionMNIST(root='../Data', train=False, download=True, transform=transform)
        class_names = ['T-shirt','Trouser','Sweater','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

        torch.manual_seed(42)
        train_loader = DataLoader(train_data,batch_size=10,shuffle=True)
        test_loader = DataLoader(test_data,batch_size=10,shuffle=False)

    def example_plot():
        
        np.set_printoptions(formatter=dict(int=lambda x: f"{x:4}"))

        for images, labels in train_loader:
            break
        Labels = labels[:10].numpy()
        print("Labels:",Labels)

        im = make_grid(images[:10],nrow=10)
        plt.figure(figsize=(10,4))
        plt.imshow(np.transpose(im.numpy(),(1,2,0)))