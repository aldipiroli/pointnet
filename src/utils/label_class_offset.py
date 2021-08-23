import os 
from os import walk 
import numpy as np
import pickle

SHAPE_NPARTS = {
    "02691156": 4,
    "02773838": 2,
    "02954340": 2,
    "02958343": 4,
    "03001627": 4,
    "03261776": 3,
    "03467517": 3,
    "03624134": 2,
    "03636649": 4,
    "03642806": 2,
    "03790512": 6,
    "03797390": 2,
    "03948459": 3,
    "04099429": 3,
    "04225987": 3,
    "04379243": 3,
}

def map_shape_label(path):
    list_dir_nparts = []
    dir_list = next(os.walk(path))[1]
    for i, dir_name in enumerate(dir_list):
        print("Processing ", i, " out of", len(dir_list))
        dir_nparts = {}

        if "0" in dir_name:
            path_label_folder = os.path.join(path, dir_name, "points_label")
            filenames = next(walk(path_label_folder), (None, None, []))[2]  # [] if no file
            max_val = -1
            for file_name in filenames:
                path_label_file = os.path.join(path_label_folder, file_name)
                labels = np.loadtxt(path_label_file, delimiter=" ", dtype=np.int32)
                max_label = np.max(labels)
                if max_label > max_val:
                    max_val = max_label

            dir_nparts[dir_name] = max_val
            list_dir_nparts.append(dir_nparts)
            print(dir_name, max_val)


    with open('map_folder_list', 'wb') as fp:
        pickle.dump(list_dir_nparts, fp)

    print("Finished saving map")

def read_map(path):
    print(path)
    with open (path, 'rb') as fp:
        itemlist = pickle.load(fp)
        print(itemlist)
    dic_ = {}
    for item in itemlist:
        for key, val in item.items():
            dic_[key] = val
    dic = {}
    for key in sorted(dic_):
        dic[key] = dic_[key]

    print("***"*20)
    print(dic)

def find_offsets():
    dic_off = {}
    i = 0
    sum_idx = 0
    for i, (key, val) in enumerate(SHAPE_NPARTS.items()):
        dic_off[key] = sum_idx
        sum_idx += val
    print(dic_off)
if __name__ == "__main__":
    # map_shape_label("/Users/aldi/workspace/pointnet/data/shape_data")
    # read_map("/Users/aldi/workspace/pointnet/src/utils/map_folder_list.pkl")
    find_offsets()