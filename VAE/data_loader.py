import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os


# use this function for each epoch
#TODO: add arg for folder and split to train, valid, test
def get_data_loader(batch_size=64, set='train', shuffle=True):
  transform = transforms.Compose(
          [transforms.RandomRotation(5),
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
           transforms.ToTensor()])

  dataset = torchvision.datasets.ImageFolder(root='../../'+set+'_data', transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return train_loader

def raw_loader(batch_size=1, set='train', shuffle=True):
  transform = transforms.ToTensor()

  dataset = torchvision.datasets.ImageFolder(root='../../'+set+'_data', transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  return train_loader
