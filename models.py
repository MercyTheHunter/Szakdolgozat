import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################
#   2D Convolutional Neural Network
##############################################################
class Conv_NN(nn.Module):
    def __init__(self):
        super(Conv_NN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1) #28 x 28 x 1 -> 24 x 24 x 6 -> 12 x 12 x 6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1) #12 x 12 x 6 -> 8 x 8 x 12 -> 4 x 4 x 12
        self.fc1 = nn.Linear(4*4*12,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        return x
    
############################################################################
#   2D Spectral Layer - Fourier Neural Operator
############################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        #Modes: Number of Fourier modes to multiply, at most Floor(N/2) + 1
        self.modes1 = modes1
        self.modes2 = modes2

        self.scale = (1 / in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2, dtype=torch.float))
    
    def compl_mul2d(self, input, weigths):
        # (batch, in_channel, x, y) -> bixy, (in_channel, out_channel, x, y) -> ioxy, (batch, out_channel, x, y) -> boxy
        return torch.einsum("bixy, ioxy->boxy", input, weigths)
    
    def forward(self, x):
        batchsize = x.shape[0]
        
        #Compute Fourier coefficients up to factor of e^(k)
        x_ft = torch.fft.rfft2(x)

        #Multiply the relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, device = x.device, dtype = torch.cfloat)
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights1))
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], torch.view_as_complex(self.weights2))
        
        #Return to pysical space
        x = torch.fft.irfft2(out_ft, s =(x.size(-2), x.size(-1)))

        return x

###########################################################################
#   2D Fourier Layer - Fourier Neural Operator
###########################################################################
class FourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, modes1, modes2):
        super(FourierLayer, self).__init__()
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                               kernel_size=kernel_size, padding=padding, stride=stride, bias=True)
        self.conv_fno2 = SpectralConv2d(in_channels=in_channels, out_channels=out_channels, modes1=modes1, modes2=modes2)

    def forward(self, x):
        x1 = self.conv2(x)
        x2 = self.conv_fno2(x)
        out = x1 + x2

        return out
    
###########################################################################
#   Neural Network Using the 2D Fourier Layer - Fourier Neural Operator
###########################################################################
class FNO_NN(nn.Module):
    def __init__(self):
        super(FNO_NN, self).__init__()
        self.fno1 = FourierLayer(in_channels=1, out_channels=32, kernel_size=5, padding=5//2, stride=1, modes1=4, modes2=4) 
        #If more than 1 Fourier Layer then the model will guess everything as 1
        self.fc1 = nn.Linear(6272,256) 
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,32)
        self.fc5 = nn.Linear(32,10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.fno1(x)
        x = F.gelu(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        #x = self.fno2(x)
        #x = F.gelu(x)
        #x = F.avg_pool2d(x, kernel_size=2, stride=2)

        #x = self.fno3(x)
        #x = F.gelu(x)
        #x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc5(x)

        return x
