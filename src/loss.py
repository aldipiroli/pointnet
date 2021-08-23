import torch
from torch import nn


class PointNetLoss(nn.Module):
    def __init__(self, device, w=0.0001):
        super(PointNetLoss, self).__init__()
        self.w = w
        self.nll_loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, gt, pr, A_):
        A = A_.clone()
        #  Orthogonality constraint
        orth = torch.norm(torch.eye(A.shape[1]).to(self.device) - torch.matmul(A, A.transpose(1, 2)))
        loss = self.nll_loss(pr, gt) + self.w * orth
        return loss


if __name__ == "__main__":
    batch_size = 5
    classes = 15

    pred = torch.randn(batch_size, classes, requires_grad=True)
    target = torch.empty(batch_size, dtype=torch.long).random_(classes)
    A = torch.rand(batch_size, 64, 64)

    print("pred.shape: ",pred.shape, "target.shape: ",target.shape, "A.shape", A.shape)
    # pred.shape:  torch.Size([5, 15]) 
    # target.shape:  torch.Size([5]) 
    # A.shape torch.Size([5, 64, 64])


    loss = PointNetLoss()
    output = loss(target, pred, A)

    print(output)
