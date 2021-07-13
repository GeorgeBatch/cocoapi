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
from torchvision import datasets, models, transforms


# can be different, defined in config.py
from config import IMG_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD

# ------------------------------------------------------------------------------------------------------------
# Strong augmentation from SimCLR (Appendix A. Data Augmentation Details)
# https://arxiv.org/pdf/2002.05709.pdf

def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2) # keeps nummber of channels
    color_distort = torch.nn.Sequential(
        rnd_color_jitter,
        rnd_gray)
    
    return color_distort

def get_gaussian_blur(img_size, ratio=0.1):
    # SimCLR: ratio=0.1
    
    # calculating kernel_size and making it odd
    kernel_size = int(img_size[0] * ratio)
    if kernel_size % 2 == 0:
        kernel_size += 1
        
    # sigma=(0.1, 2.0) is default for GaussianBlur; it is also the same in Sim CLR
    gaussian_blur = transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))
    
    # SimCLR: p=0.5 
    rnd_gaussian_blur = transforms.RandomApply([gaussian_blur], p=0.5)
    
    return rnd_gaussian_blur


# ------------------------------------------------------------------------------------------------------------
# define the transforms

# only add flipping
baseline_train_transforms = torch.nn.Sequential(
    transforms.Resize(size=IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
)

# Strong augmentation from SimCLR
SimCLR_train_transforms = torch.nn.Sequential(
    transforms.Resize(size=IMG_SIZE), # different from SimCLR
    # SimCLR aug start
    transforms.RandomHorizontalFlip(),
    get_color_distortion(s=1.0),
    get_gaussian_blur(IMG_SIZE),
    # SimCLR aug end
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
)

# only need to get the images to the correct format
dev_transforms = torch.nn.Sequential(
    transforms.Resize(size=IMG_SIZE),
    transforms.Normalize(NORMALIZATION_MEAN, NORMALIZATION_STD),
)