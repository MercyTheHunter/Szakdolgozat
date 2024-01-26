import torch.nn as nn
import torch.nn.functional as F

class Conv_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1) #28 x 28 x 1 -> 24 x 24 x 6 -> 12 x 12 x 6
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1) #12 x 12 x 6 -> 8 x 8 x 12 -> 4 x 4 x 12
        self.fc1 = nn.Linear(4*4*12,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)

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
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x