from torch.utils.data import Dataset, DataLoader
import os
import re


class ShapeNetDatasetClass(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        assert os.path.isdir(self.data_path), ("Data Path is Not Corret: ", self.data_path)

        self.class_map = self.load_class_map()
        files = self.load_files()
        print(files)

    def load_files(self):
        folders = [x[0] for x in os.walk(self.data_path)]
        return folders


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
            key, val = re.split(r"\t+", content)
            class_map[key] = val

        return class_map

    def __len__(self):

        return 0

    def __getitem__(self, idx):
        return 0


if __name__ == "__main__":
    data = ShapeNetDatasetClass("/Users/aldi/workspace/pointnet/data/")
    # dataloader = DataLoader(data, batch_size=4, shuffle=True, num_workers=0)
