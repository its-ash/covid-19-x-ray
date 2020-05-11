import importlib
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, utils , models
from dataset import dataset_path, create_dataset


WIDTH = 300
HEIGHT = 300


transformation = transforms.Compose([
    transforms.Resize((WIDTH+50, HEIGHT+50)),
    transforms.CenterCrop((WIDTH, HEIGHT)),
    transforms.Grayscale(),
    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1,
    ),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.456],
                         std=[0.224]),
   
])


create_dataset()

dataset = datasets.ImageFolder(dataset_path, transform=transformation)

def get_loader(BATCH_SIZE=64):
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def plot_sample_data():
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    images,_ = next(iter(dataloader))
    plt.imshow(utils.make_grid(images)[0])

