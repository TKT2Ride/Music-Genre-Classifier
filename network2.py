import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets


class Genre_Model(nn.Module):

    def __init__(self):
        super().__init__()
        # Input images are going to be 432 x 288
        # batch * out_channels * 432 * 288

        # NOTES: Up the number of channels at every layer
        # Ex: Start with 32 and go up by powers of 2?
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        # batch * out_channels * 214 * 142
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(6)
        # batch * out_channels * 105 * 69
        self.conv3 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(6)
        # batch * out_channels * 50 * 32
        self.conv4 = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(6)
        # batch * out_channels * 23 * 14
        self.conv5 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(16)

        self.dropout = nn.Dropout()
        # Out channels by the dimensions of the previous layer
        self.fc1 = nn.Linear(480, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 10)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # print(0, x.size())
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # print(1, x.size())
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # print(2, x.size())
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # print(3, x.size())
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # print(4, x.size())
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        # print(5, x.size())

        x = self.dropout(x)

        # print('dropout', x.size())
        x = torch.flatten(x, 1)  # flatten all dimensions except batch

        # print('flatten', x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = self.softmax(x)
        # print(x.size())
        return x
