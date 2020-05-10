
import os
import glob
from random import choice
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils , models
from pathlib import Path
from 

WIDTH = 256
HEIGHT = 256


transformation = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.Grayscale(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
    ),
    transforms.RandomRotation(10),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.ToTensor()
])



from Net import CustomNet

model = CustomNet(3)