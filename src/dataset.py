from torch.utils.data import Dataset, DataLoader
import os
import re
import json
import torch 
import numpy as np

class ShapeNetDataset(Dataset):
    def __init__(self, data_path, N=1024, augment=False):
        """
        N: number of points to sample out of the total shape
        """
        self.data_path = data_path
        assert os.path.isdir(self.data_path), ("Data Path is Not Corret: ", self.data_path)
        self.N = N
        self.augment = augment

        self.class_map = self.load_class_map()
        self.file_list = self.load_files(2)

    def load_files(self, split):
        """ 
        split 0: test
        split 1: train
        split 2: val
        """

        split_name = [
            "shuffled_test_file_list.json",
            "shuffled_train_file_list.json",
            "shuffled_val_file_list.json",
        ]
        path = os.path.join(self.data_path, "shape_data", "train_test_split", split_name[split])
        assert os.path.isfile(path), ("Path does not exist", path)

        datas = []
        with open(path) as json_file:
            data_list = json.load(json_file)
            for data in data_list:
                d = re.split(r"/+", data)

                data = {}
                points_ = os.path.join(self.data_path, d[0], d[1], "points", d[2] + ".pts")
                label_ = os.path.join(self.data_path, d[0], d[1], "points_label", d[2] + ".seg")

                assert os.path.isfile(points_), ("Points file does not exist: ", points_)
                assert os.path.isfile(label_), ("Label file does not exist: ", label_)
                data["points"] = points_
                data["label"] = label_
                data["class"] = self.class_map[d[1]]
                datas.append(data)


        return datas

    def load_class_map(self):
        """ 
        Create a map label -> folder_name
        """
        path = os.path.join(self.data_path, "shape_data/synsetoffset2category.txt")
        assert os.path.isfile(path), ("The file does not exist: ", path)

        f = open(path, "r")
        content = f.read()
        content_list = content.splitlines()
        f.close()

        class_map = {}
        for content in content_list:
            label, folder = re.split(r"\t+", content)
            class_map[folder] = label

        return class_map

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = self.file_list[idx]
        points = np.loadtxt(data["points"], delimiter = " ", dtype=np.float32)
        labels = np.loadtxt(data["label"], delimiter = " ", dtype=np.int32)
        class_name = data["class"]




        return data


if __name__ == "__main__":
    data = ShapeNetDataset("/Users/aldi/workspace/pointnet/data/")
    data[0]
    # dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0)
