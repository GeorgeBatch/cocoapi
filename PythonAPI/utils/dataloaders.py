# A lot of the code is from here:
# * https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# * https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

import os

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets
from torchvision.io import read_image # reads image straingt to torch.tensor compared to 
from torchvision.transforms import ToTensor




class CocoNoCropping(Dataset):
    def __init__(self, img_ids, my_annotations_file, img_dir,
                 transform=None, target_transform=None, divide_by_255=False):

        # dataDir, dataType, annFile, 
        # self.coco_datatype = coco_datatype
        # self.coco_annotations_file = 
        
        with open(img_ids, 'r') as f:
            self.img_ids = json.loads(f.read()) # python list saved as txt IDs as ints
        with open(my_annotations_file, 'r') as f:
            self.ids_to_labels = json.load(f)   # python dictionary saved as json IDs as strings
            
        self.img_dir = img_dir
        
        # target transforms
        self.target_transform = target_transform
        # image transforms
        self.transform = transform
        self.divide_by_255 = divide_by_255
        

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = str(self.img_ids[idx])
        img_name = (12-len(img_id)) * '0' + img_id + '.jpg'
        
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        
        # for b/w images just stack the same channel together 3 times
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        
        if self.divide_by_255:
            image = image / 255.0
        
        label = self.ids_to_labels[img_id]
        if self.transform:
            image = self.transform(image)
            
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample