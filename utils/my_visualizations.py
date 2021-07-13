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

def imshow_from_normalized(inp, title=None):
    """Imshow for normalized Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    print(inp.shape)
    
    # ImageNet constants
    mean = np.array(NORMALIZATION_MEAN)
    std = np.array(NORMALIZATION_STD)
    
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated