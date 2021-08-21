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

        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(F.relu(self.bn5(self.fc2(x))))

        x = self.fc3(x)

        iden = torch.eye(3, 3).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.to(self.device)

        x = x.view(-1, 3, 3)
        x = x + iden

        return x


class TNet64(nn.Module):
    def __init__(self, device):
        super(TNet64, self).__init__()

        self.device = device
        self.K = 64

        self.conv1 = nn.Conv1d(64, 64, 1)
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

        self.dropout = nn.Dropout(p=0.7)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout(F.relu(self.bn5(self.fc2(x))))
        x = self.fc3(x)

        iden = torch.eye(self.K, self.K).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.to(self.device)

        x = x.view(-1, self.K, self.K)
        x = x + iden

        return x


class PointNetClass(nn.Module):
    def __init__(self, device, k=10):
        """
        k: number of classes which a the input (shape) can be classified into
        """
        super(PointNetClass, self).__init__()
        self.device = device
        self.k = k

        self.TNet3 = TNet3(self.device)
        self.TNet64 = TNet64(self.device)

        self.mlp1 = nn.Conv1d(3, 64, 1)
        self.mlp2 = nn.Conv1d(64, 64, 1)
        self.mlp3 = nn.Conv1d(64, 64, 1)
        self.mlp4 = nn.Conv1d(64, 128, 1)
        self.mlp5 = nn.Conv1d(128, 1024, 1)

        self.mlp6 = nn.Conv1d(1024, 512, 1)
        self.mlp7 = nn.Conv1d(512, 256, 1)
        self.mlp8 = nn.Conv1d(256, self.k, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        #  input transform:
        x_ = x.clone()
        T3 = self.TNet3(x_)
        x = torch.matmul(T3, x)

        #  mlp (64,64):
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))

        # feature transform:
        x_ = x.clone()
        T64 = self.TNet64(x_)
        print((x_.shape, x.shape))
        x = torch.matmul(T64, x)

        #  mlp (64,128,1024):
        x = F.relu(self.bn3(self.mlp3(x)))
        x = F.relu(self.bn4(self.mlp4(x)))
        x = F.relu(self.bn5(self.mlp5(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn6(self.mlp6(x)))
        x = F.relu(self.bn7(self.dropout(self.mlp7(x))))
        x = self.mlp8(x)

        return x.squeeze()


class PointNetSeg(nn.Module):
    def __init__(self, device, m=50):
        """
        m: number of classes which a single point can be classified into
        """
        super(PointNetSeg, self).__init__()
        self.device = device
        self.m = m

        self.TNet3 = TNet3(self.device)
        self.TNet64 = TNet64(self.device)

        self.mlp1 = nn.Conv1d(3, 64, 1)
        self.mlp2 = nn.Conv1d(64, 64, 1)
        self.mlp3 = nn.Conv1d(64, 64, 1)
        self.mlp4 = nn.Conv1d(64, 128, 1)
        self.mlp5 = nn.Conv1d(128, 1024, 1)

        # segmentation part
        self.mlp6 = nn.Conv1d(1088, 512, 1)
        self.mlp7 = nn.Conv1d(512, 256, 1)
        self.mlp8 = nn.Conv1d(256, 128, 1)
        self.mlp9 = nn.Conv1d(128, self.m, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)


        # segmentation part:
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)


        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        #  input transform:
        x_ = x.clone()
        T3 = self.TNet3(x_)
        x = torch.matmul(T3, x)

        #  mlp (64,64):
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))

        # feature transform:
        x_ = x.clone()
        T64 = self.TNet64(x_)
        print((x_.shape, x.shape))
        x = torch.matmul(T64, x)

        x_feature = x.clone()

        #  mlp (64,128,1024):
        x = F.relu(self.bn3(self.mlp3(x)))
        x = F.relu(self.bn4(self.mlp4(x)))
        x = F.relu(self.bn5(self.mlp5(x)))

        x_globfeat = torch.max(x, 2, keepdim=True)[0]

        # Concatenate global and local features
        x_globfeat = x_globfeat.expand(-1, -1, x_feature.shape[2])
        x = torch.cat((x_feature, x_globfeat), dim=1)

        x = F.relu(self.bn6(self.mlp6(x)))
        x = F.relu(self.bn7(self.mlp7(x)))
        x = F.relu(self.bn8(self.mlp8(x)))

        x = self.mlp9(x)


        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input = torch.rand((10, 3, 500))
    net = TNet3(device)
    y = net(input)
    print("T-Net 3", y.shape)

    input = torch.rand((10, 64, 500))
    net = TNet64(device)
    y = net(input)
    print("T-Net 64", y.shape)

    input = torch.rand((10, 3, 500))
    net = PointNetClass(device, 15)
    y = net(input)
    print("PointNet Class", y.shape)

    input = torch.rand((10, 3, 500))
    net = PointNetSeg(device, 15)
    x = net(input)
    print("PointNet Segm", x.shape)
