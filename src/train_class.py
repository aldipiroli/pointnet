from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim

import numpy as np
from dataset import ShapeNetDataset
from network import PointNetClass
from loss import PointNetLoss


class Trainer():
    def __init__(self):
        # Set Training Param:
        self.batch_size = 2
        self.lr = 0.001
        self.n_epochs = 1000

        # Use GPU?
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Dataloader:
        self.dataset = ShapeNetDataset("/Users/aldi/workspace/pointnet/data/", augment=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        # Network:
        self.net = PointNetClass(self.device, classes=16)

        # Optimizer:
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # Loss:
        self.loss = PointNetLoss()

    def train(self):
        for epoch in range(self.n_epochs):
            print("\n============= Epoch: %d =============\n" % epoch)
            for i, (points, _, target) in enumerate(self.dataloader):
                # Compute Network Output
                pred, A = self.net(points)

                # Compute Loss
                loss = self.loss(target, pred, A)

                # Optimize:
                self.optimizer.zero_grad()  
                loss.backward()
                self.optimizer.step()

                print("loss: ", loss)
                input("..")

    def train_overfit(self):
        self.net.train()
        points, _, target = next(iter(self.dataloader))

        for epoch in range(self.n_epochs):
            print("\n============= Epoch: %d =============\n" % epoch)

            # Compute Network Output
            pred, A = self.net(points)

            # Compute Loss
            loss = self.loss(target, pred, A)

            # Optimize:
            self.optimizer.zero_grad()  
            loss.backward()
            self.optimizer.step()

            # Compute readable pred:
            pred_ = torch.max(pred, dim=1)

            
            if epoch % 20 == 0:
                print("loss: ", loss)
                print("target: ", target)
                print("pred: ", pred)
                print("pred_: ", pred_)
                input("..")

        
if __name__ == "__main__":
    trainer = Trainer()
    # trainer.train()
    trainer.train_overfit()
