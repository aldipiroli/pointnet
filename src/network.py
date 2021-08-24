import torch
from torch import nn
import torch.nn.functional as F


class mlp(nn.Module):
    def __init__(self, in_size, out_size, k_size=1, batchnorm=True):
        """ 
        Creates a mlp layer as described in the paper.

        in_size: input size of the mlp
        out_size: output size of the mlp
        relu: apply relu
        batchnorm: apply norm 
        """
        super(mlp, self).__init__()
        self.batchnorm = batchnorm
        self.conv = nn.Conv1d(in_size, out_size, k_size)
        self.bn = nn.BatchNorm1d(out_size)

    def forward(self, x):
        if self.batchnorm:
            return F.relu(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class fc(nn.Module):
    def __init__(self, in_size, out_size, k_size=1, batchnorm=True, dropout=False, dropout_p=0.7):
        """ 
        Creates a fully connected layer as described in the paper.

        in_size: input size of the mlp
        out_size: output size of the mlp
        relu: apply relu
        batchnorm: apply norm 
        """
        super(fc, self).__init__()
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.fc = nn.Linear(in_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.dp = nn.Dropout(p=dropout_p)

    def forward(self, x):
        if self.batchnorm and not self.dropout:
            return F.relu(self.bn(self.fc(x)))
        elif self.batchnorm and self.dropout:
            return F.relu(self.bn(self.dp(self.fc(x))))
        elif not self.batchnorm:
            return self.fc(x)


class TNet3(nn.Module):
    def __init__(self, device):
        super(TNet3, self).__init__()

        self.device = device

        self.mlp1 = mlp(3, 64)
        self.mlp2 = mlp(64, 128)
        self.mlp3 = mlp(128, 1024)

        self.fc1 = fc(1024, 512)
        self.fc2 = fc(512, 256, dropout=True)
        self.fc3 = fc(256, 9)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        x = torch.max(x, 2)[0]

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        #  For stability
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

        self.mlp1 = mlp(64, 64)
        self.mlp2 = mlp(64, 128)
        self.mlp3 = mlp(128, 1024)

        self.fc1 = fc(1024, 512)
        self.fc2 = fc(512, 256, dropout=True)
        self.fc3 = fc(256, 64 * 64)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)

        x = torch.max(x, 2)[0]

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        #  For stability
        iden = torch.eye(64, 64).repeat(batch_size, 1, 1)
        if x.is_cuda:
            iden = iden.to(self.device)

        x = x.view(-1, 64, 64)
        x = x + iden

        return x


class PointNetClass(nn.Module):
    def __init__(self, device, classes=10):
        """
        k: number of classes which a the input (shape) can be classified into
        """
        super(PointNetClass, self).__init__()
        self.device = device
        self.classes = classes

        self.TNet3 = TNet3(self.device)
        self.TNet64 = TNet64(self.device)

        self.mlp1 = mlp(3, 64)
        self.mlp2 = mlp(64, 64)
        self.mlp3 = mlp(64, 128)
        self.mlp4 = mlp(128, 1024)

        self.mlp5 = mlp(1024, 512)
        self.mlp6 = mlp(512, 256)
        self.mlp7 = mlp(256, self.classes, batchnorm=False)

    def forward(self, x):
        #  input transform:
        x_ = x.clone()
        T3 = self.TNet3(x_)
        x = torch.matmul(T3, x)

        #  mlp (64,64):
        x = self.mlp1(x)
        x = self.mlp2(x)

        # feature transform:
        x_ = x.clone()
        T64 = self.TNet64(x_)
        x = torch.matmul(T64, x)

        #  mlp (64,128,1024):
        x = self.mlp3(x)
        x = self.mlp4(x)

        x = torch.max(x, 2, keepdim=True)[0]

        x = self.mlp5(x)
        x = self.mlp6(x)
        x = self.mlp7(x)

        return x.squeeze(), T64


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

        self.mlp1 = mlp(3, 64)
        self.mlp2 = mlp(64, 64)
        self.mlp3 = mlp(64, 64)
        self.mlp4 = mlp(64, 128)
        self.mlp5 = mlp(128, 1024)

        self.mlp6 = mlp(1088, 512)
        self.mlp7 = mlp(512, 256)
        self.mlp8 = mlp(256, 128)
        self.mlp9 = mlp(128, self.m, batchnorm=False)


    def forward(self, x):
        #  input transform:
        x_ = x.clone()
        T3 = self.TNet3(x_)
        x = torch.matmul(T3, x)

        #  mlp (64,64):
        x = self.mlp1(x)
        x = self.mlp2(x)

        # feature transform:
        x_ = x.clone()
        T64 = self.TNet64(x_)

        x = torch.matmul(T64, x)

        x_feature = x.clone()

        #  mlp (64,128,1024):
        x = self.mlp3(x)
        x = self.mlp4(x)
        x = self.mlp5(x)

        x_globfeat = torch.max(x, 2, keepdim=True)[0]

        #  Concatenate global and local features
        x_globfeat = x_globfeat.expand(-1, -1, x_feature.shape[2])
        x = torch.cat((x_feature, x_globfeat), dim=1)

        x = self.mlp6(x)
        x = self.mlp7(x)
        x = self.mlp8(x)
        x = self.mlp9(x)

        return x, T64


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
    net = PointNetSeg(device, 50)
    x = net(input)
    print("PointNet Segm", x.shape)
