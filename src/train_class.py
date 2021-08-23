from torch.utils.data import Dataset, DataLoader
import torch
import torch.optim as optim
import os 

import numpy as np
from dataset import ShapeNetDataset
from network import PointNetClass
from loss import PointNetLoss


class Trainer:
    def __init__(self):
        DATASET_PATH = "/content/drive/MyDrive/data/"
        DATASET_PATH = "data/"

        # Training Parameters:
        self.batch_size = 16
        self.lr = 0.001
        self.n_epochs = 1000
        self.model_path = "model/model.pth"
        self.load_model= True

        # Use GPU?
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Training on Device: ", self.device)

        # ===== Dataloader =====:        
        self.dataset = ShapeNetDataset(DATASET_PATH, augment=True, split=1)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        self.dataset_val = ShapeNetDataset(DATASET_PATH, augment=True, split=2)
        self.dataloader_val = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False)

        #  Network:
        self.net = PointNetClass(self.device, classes=16).to(self.device)

        # Optimizer:
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        # Loss:
        self.loss = PointNetLoss(self.device)

        # Load Model?
        if self.load_model and os.path.isfile(self.model_path):
            self.net = torch.load(self.model_path)
            print("Loaded Path: ", self.model_path)

    def train(self):
        for epoch in range(self.n_epochs):
            print("\n============= Epoch: %d =============\n" % epoch)
            print("Len: ", len(self.dataloader))
            #  Training Loop:
            self.net.train()
            for i, (points, _, target) in enumerate(self.dataloader):
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
                    print("\nEpoch: %d, i: %d, Error Loss: %f" % (epoch, i, loss))
                    pred_ = torch.max(pred,1)[1]
                    print("Pred: ", pred_)
                    print("Targ: ", target)

            # Validate:
            self.net.eval()
            val_loss = 0
            for i, (points, _, target) in enumerate(self.dataloader_val):
                points = points.to(self.device)
                target = target.to(self.device)

                pred, A = self.net(points)
                loss = self.loss(target, pred, A)
                val_loss += loss
            print("Epoch: %d, i: %d, Validation Loss: %f" % (epoch, i, val_loss))

            # Save the model:
            torch.save(self.net.state_dict(), self.model_path)


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

            #  Compute readable pred:
            pred_ = torch.max(pred, dim=1)

            if epoch % 20 == 0:
                print("loss: ", loss)
                print("target: ", target)
                print("pred: ", pred)
                print("pred_: ", pred_)
                input("..")


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    # trainer.train_overfit()
