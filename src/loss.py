import torch
from torch import nn


class PointNetLoss(nn.Module):
    def __init__(self, w=0.0001):
        super(PointNetLoss, self).__init__()
        self.w = w
        self.nll_loss = nn.CrossEntropyLoss()

    def forward(self, gt, pr, A):
        #  Orthogonality constraint
        orth = torch.norm(torch.eye(A.shape[0]) - torch.matmul(A, A.transpose(0, 1)))

        return self.nll_loss(pr, gt) + self.w * orth


if __name__ == "__main__":
    batch_size = 10
    classes = 15

    pr = torch.randn(batch_size, classes, requires_grad=True)
    gt = torch.empty(batch_size, dtype=torch.long).random_(classes)
    A = torch.rand(64, 64)

    loss = PointNetLoss()
    output = loss(gt, pr, A)

    print(output)
