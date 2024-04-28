import torch
import torch.nn as nn
import torch.nn.functional as F

##############################################################
#   2D Convolutional Neural Networks
##############################################################
class Conv_NN_small(nn.Module):
    def __init__(self, kernelsize, classes, in_channels):
        super(Conv_NN_small, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=kernelsize, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = self.ap(x)
        
        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        return x
class Conv_NN_medium(nn.Module):
    def __init__(self, kernelsize, classes, in_channels):
        super(Conv_NN_medium, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=kernelsize, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernelsize, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,classes)


    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = self.ap(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        return x
class Conv_NN_big(nn.Module):
    def __init__(self, kernelsize, classes, in_channels):
        super(Conv_NN_big, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=kernelsize, stride=1)
        self.bn1 = nn.BatchNorm2d(num_features=2)

        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=kernelsize, stride=1)
        self.bn2 = nn.BatchNorm2d(num_features=4)

        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=kernelsize, stride=1)
        self.bn3 = nn.BatchNorm2d(num_features=8)

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernelsize, stride=1)
        self.bn4 = nn.BatchNorm2d(num_features=16)

        self.conv5 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernelsize, stride=1)
        self.bn5 = nn.BatchNorm2d(num_features=32)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv1(x)
        x = F.gelu(x)
        x = self.bn1(x)
        
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = F.gelu(x)
        x = self.bn2(x)
        
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3(x)
        x = F.gelu(x)
        x = self.bn3(x)
        
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv4(x)
        x = F.gelu(x)
        x = self.bn4(x)
        
        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.conv5(x)
        x = F.gelu(x)
        x = self.bn5(x)
        
        x = self.ap(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)

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
#   Neural Networks Using the 2D Fourier Layer - Fourier Neural Operator
###########################################################################
class FNO_NN_small(nn.Module):
    def __init__(self, kernelsize, classes, in_channels):
        super(FNO_NN_small, self).__init__()
        self.fno1 = FourierLayer(in_channels=in_channels, out_channels=32, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4) 
        self.bn1 = nn.BatchNorm2d(num_features=32)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.fno1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = self.ap(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        return x
class FNO_NN_medium(nn.Module):
    def __init__(self, kernelsize, classes, in_channels):
        super(FNO_NN_medium, self).__init__()
        self.fno1 = FourierLayer(in_channels=in_channels, out_channels=16, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4)
        self.bn1 = nn.BatchNorm2d(num_features=16)

        self.fno2 = FourierLayer(in_channels=16, out_channels=32, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4)
        self.bn2 = nn.BatchNorm2d(num_features=32)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.fno1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.fno2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = self.ap(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        return x
class FNO_NN_big(nn.Module):
    def __init__(self, kernelsize, classes, in_channels):
        super(FNO_NN_big, self).__init__()
        self.fno1 = FourierLayer(in_channels=in_channels, out_channels=2, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4)
        self.bn1 = nn.BatchNorm2d(num_features=2)

        self.fno2 = FourierLayer(in_channels=2, out_channels=4, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4)
        self.bn2 = nn.BatchNorm2d(num_features=4)

        self.fno3 = FourierLayer(in_channels=4, out_channels=8, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4)
        self.bn3 = nn.BatchNorm2d(num_features=8)

        self.fno4 = FourierLayer(in_channels=8, out_channels=16, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4)
        self.bn4 = nn.BatchNorm2d(num_features=16)

        self.fno5 = FourierLayer(in_channels=16, out_channels=32, kernel_size=kernelsize, padding=kernelsize//2, stride=1, modes1=4, modes2=4)
        self.bn5 = nn.BatchNorm2d(num_features=32)

        self.ap = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(32,128)
        self.fc2 = nn.Linear(128,classes)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.fno1(x)
        x = F.gelu(x)
        x = self.bn1(x)

        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.fno2(x)
        x = F.gelu(x)
        x = self.bn2(x)

        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.fno3(x)
        x = F.gelu(x)
        x = self.bn3(x)

        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.fno4(x)
        x = F.gelu(x)
        x = self.bn4(x)

        x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.fno5(x)
        x = F.gelu(x)
        x = self.bn5(x)

        x = self.ap(x)

        x = x.view(batch_size, -1)

        x = self.fc1(x)
        x = F.gelu(x)

        x = self.fc2(x)

        return x