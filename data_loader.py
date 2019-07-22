import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os

# we might need to use this in the future 
# for now ignore it
# pip install split-folders
# import split_folders
# Split with a ratio.
# To only split into training and validation set, set a tuple to `ratio`, i.e, `(.8, .2)`.
# split_folders.ratio('./face_data', output="./output", seed=1337, ratio=(.8, .2)) # default values

# use this function for each epoch
def get_data_loader(batch_size=64):
  transform = transforms.Compose(
          [transforms.RandomRotation(5),
           transforms.RandomHorizontalFlip(p=0.5),
           transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
           transforms.ToTensor(),
           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  dataset = torchvision.datasets.ImageFolder(root='./face_data', transform=transform)
  train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=1, shuffle=True)
  print(train_loader)
  return train_loader


if __name__ == '__main__':
    # it = DataLoader([1,2,3,45,5], batch_size=2, num_workers=1)
    # print('started')
    # for i in it:
    #     print(i)

    train_loader = get_data_loader(batch_size=1)

    k = 0
    for images, label in train_loader:
        print(images.shape)
        image = images[0]
        # place the colour channel at the end, instead of at the beginning
        img = np.transpose(image, [1,2,0])
        # normalize pixel intensity values to [0, 1]
        img = img / 2 + 0.5
        plt.subplot(3, 5, k+1)
        plt.axis('off')
        plt.imshow(img)
        k += 1
        if k > 14:
           break

    plt.show()
