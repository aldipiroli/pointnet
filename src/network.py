import torch
from torch import nn
import torch.nn.functional as F


class TNet3(nn.Module):
    def __init__(self, device):
        super(TNet3, self).__init__()

        self.device = device

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, 3).repeat(batch_size,1, 1)
        if x.is_cuda:
          iden = iden.to(self.device)

        x = x.view(-1, 3, 3)
        x = x + iden


        return x, iden


class TNet64(nn.Module):
    def __init__(self, device):
        super(TNet64, self).__init__()

        self.device = device
        self.K = 64

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4096)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.K, self.K).repeat(batch_size,1, 1)
        if x.is_cuda:
          iden = iden.to(self.device)

        x = x.view(-1, self.K, self.K)
        x = x + iden


        return x, iden


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input = torch.rand((10, 3, 500))
    net = TNet3(device)
    y, iden = net(input)
    print(y.shape)

    input = torch.rand((10, 3, 500))
    net = TNet64(device)
    y, iden = net(input)
    print(y.shape)