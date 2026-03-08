import torch
import torch.nn as nn


class APAN(nn.Module):

    def __init__(self):
        super().__init__()

        self.a = nn.Parameter(torch.tensor(0.5))
        self.b = nn.Parameter(torch.tensor(0.5))
        self.c = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return self.a * x**2 + self.b * x + self.c


class BrainTumorCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)

        self.pool = nn.AvgPool2d(2)

        self.apan = APAN()

        self.fc1 = nn.Linear(32*32*32,128)
        self.fc2 = nn.Linear(128,4)

    def forward(self,x):

        x = self.conv1(x)
        x = self.apan(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.apan(x)
        x = self.pool(x)

        x = x.view(x.size(0),-1)

        x = self.fc1(x)
        x = self.apan(x)

        x = self.fc2(x)

        return x