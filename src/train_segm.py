from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import os 
from torch import nn

import numpy as np
from dataset import ShapeNetDataset
from network import PointNetClass, PointNetSeg
from loss import PointNetLoss


class Trainer:
    def __init__(self):
        self.DATASET_PATH = "/content/drive/MyDrive/data/"
        self.DATASET_PATH = "data/"

        # Training Parameters:
        self.batch_size = 2
        self.lr = 0.001
        self.n_epochs = 1000
        self.model_path = "model/model_segm.pth"
        self.load_model= False
        self.compute_validation = False

        # Use GPU?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training on Device: ", self.device)

        # ===== Dataloader =====:        
        self.dataset = ShapeNetDataset(self.DATASET_PATH, augment=True, split=1)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        self.dataset_val = ShapeNetDataset(self.DATASET_PATH, augment=True, split=2)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)

        #  Network:
        self.net = PointNetSeg(self.device).to(self.device)

        # Optimizer:
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # Loss:
        self.loss = PointNetLoss(self.device)

        # Load Model?
        if self.load_model and os.path.isfile(self.model_path):
            self.net = torch.load(self.model_path)
            print("Loaded Path: ", self.model_path)

    def train(self):
        print("Training Segmentation PointNet!")
        points, target, _ = next(iter(self.dataloader))
        i = 1

        for epoch in range(self.n_epochs):
            #  Training Loop:
            self.net.train()
            for i, (points, target, _) in enumerate(self.dataloader):
                points = points.to(self.device)
                target = target.to(self.device)

                # Compute Network Output
                pred, A = self.net(points)

                # Compute Loss
                loss = self.loss(target, pred, A)

                # Optimize:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if i % 25 == 0:
                    print("="*50)
                    print("Epoch: %d, i: %d, Error Loss: %f" % (epoch, i, loss))
                    pred_ = torch.max(pred,1)[1]
                    print("Pred: ", pred_)
                    print("Targ: ", target)
            
            # Save the model:
            torch.save(self.net.state_dict(), self.model_path)

            # Validate:
            self.net.eval()
            val_loss = 0
            if self.compute_validation:
                for i, (points, target,_) in enumerate(self.dataloader_val):
                    points = points.to(self.device)
                    target = target.to(self.device)

                    pred, A = self.net(points)
                    loss = self.loss(target, pred, A)
                    val_loss += loss
                    if i % 25 == 0:
                        print("Epoch: %d, i: %d, Validation Loss: %f" % (epoch, i, val_loss))
    def mIoU(self):
        dataset = ShapeNetDataset(self.DATASET_PATH, augment=True, split=0)
        dataloader = DataLoader(self.dataset_val, batch_size=2, shuffle=False)

        for i, (points, target, _) in enumerate(dataloader):
            points = points.to(self.device)
            target = target.to(self.device)

            pred, _ = self.net(points)

            # Find arg max of prediction:
            max_ = torch.max(pred,1)[1]

            print("Pred: ", pred.shape)
            print("Max: ", max_.shape)
            print("Target: ", target.shape)
            input("...")

if __name__ == "__main__":
    trainer = Trainer()
    # trainer.train()
    trainer.mIoU()
