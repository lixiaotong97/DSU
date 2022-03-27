import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import pickle
import imageio


class synthiaDataSet(data.Dataset):
    def __init__(
        self,
        data_root,
        data_list,
        max_iters=None,
        num_classes=16, 
        split="train",
        transform=None,
        ignore_label=255,
        debug=False,
    ):
        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]
        
        if max_iters is not None:
            self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, "synthia_label_info.p"), "rb"))
            self.img_ids = []
            SUB_EPOCH_SIZE = 3000
            tmp_list = []
            ind = dict()
            for i in range(self.NUM_CLASS):
                ind[i] = 0
            for e in range(int(max_iters/SUB_EPOCH_SIZE)+1):
                cur_class_dist = np.zeros(self.NUM_CLASS)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist/cur_class_dist.sum()
                    w = 1/np.log(1+1e-2 + dist1)
                    w = w/w.sum()
                    c = np.random.choice(self.NUM_CLASS, p=w)

                    if ind[c] > (len(self.label_to_file[c])-1):
                        np.random.shuffle(self.label_to_file[c])
                        ind[c] = ind[c]%(len(self.label_to_file[c])-1)

                    c_file = self.label_to_file[c][ind[c]]
                    tmp_list.append(c_file)
                    ind[c] = ind[c]+1
                    cur_class_dist[self.file_to_label[c_file]] += 1

            self.img_ids = tmp_list

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, "RAND_CITYSCAPES/RGB/%s" % name),
                    "label": os.path.join(self.data_root, "RAND_CITYSCAPES/GT/LABELS//%s" % name),
                    "name": name,
                }
            )
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))

        self.id_to_trainid = {
            3: 0,
            4: 1,
            2: 2,
            21: 3,
            5: 4,
            7: 5,
            15: 6,
            9: 7,
            6: 8,
            1: 9,
            10: 10,
            17: 11,
            8: 12,
            19: 13,
            12: 14,
            11: 15,
        }
        self.trainid2name = {
            0: "road",
            1: "sidewalk",
            2: "building",
            3: "wall",
            4: "fence",
            5: "pole",
            6: "light",
            7: "sign",
            8: "vegetation",
            9: "sky",
            10: "person",
            11: "rider",
            12: "car",
            13: "bus",
            14: "motocycle",
            15: "bicycle",
        }
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        # label = np.array(Image.open(datafiles["label"]),dtype=np.float32)
        label = np.asarray(imageio.imread(datafiles["label"], format='PNG-FI'))[:,:,0]  # uint16
        name = datafiles["name"]

        # re-assign labels to match the format of Cityscapes
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, name
