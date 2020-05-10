import importlib
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils , models
from pathlib import Path


WIDTH = 300
HEIGHT = 300


transformation = transforms.Compose([
    transforms.Resize((WIDTH, HEIGHT)),
    transforms.Grayscale(),
    transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
    ),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456],
                         std=[0.224]),
   
])



dataset = datasets.ImageFolder('dataset', transform=transformation)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

images,labels = next(iter(dataloader))
plt.imshow(utils.make_grid(images)[0])

