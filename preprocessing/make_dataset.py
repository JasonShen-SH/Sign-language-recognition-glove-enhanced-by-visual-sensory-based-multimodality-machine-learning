import torch
import torch.nn as nn
import pdb

import os
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import random_split, Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
import random

# make dataset
class GestureDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'label': self.labels[idx]
        }
        return sample

    
# load data
img_data = torch.load('/mnt/fyp/data/img_data.pt').reshape(3995,-1)
teng_data = torch.load('/mnt/fyp/data/teng_data.pt')
imu_data = torch.load('/mnt/fyp/data/imu_data.pt')
count = torch.load('/mnt/fyp/data/count.pt')
# img_data: 3995*(3*224*224)
# teng_data: 3995*(50*10)
# imu_data: 3995*(50*18)


# merge data
merged_data = torch.cat((img_data, teng_data, imu_data),dim=1) # must be 3995*151928


# prune merged data to 100 samples per class
for i,num in enumerate(count):
    if i==0:
        pruned_merged_data = merged_data[:min(num,100), :]
    else:
        pruned_merged_data = torch.cat((pruned_merged_data, merged_data[:min(num,100), :]), dim=0)
        
    merged_data = merged_data[max(num, 100):, :] 
# pruned_merged_data must be 3900*151928


# random train-test(80%-20%) split for each category
class_data = [[] for _ in range(39)]
pruned_labels = torch.tensor([i for i in range(39) for _ in range(100)]).reshape(-1,1)
for class_idx in range(39):
    class_mask = (pruned_labels == class_idx).squeeze()
    data = pruned_merged_data[class_mask]
    labels = pruned_labels[class_mask]
    
    train_size = int(0.8 * len(data)) # must be 80
    val_size = len(data) - train_size # must be 20
    
    # split with random indices
    num_samples = len(data) # must be 100
    random_indices = torch.randperm(num_samples)
    train_data = data[random_indices[:train_size]]
    val_data = data[random_indices[train_size:]]
    
    class_data[class_idx] = (train_data, val_data)
    
    
# form train & val dataset
for i in range(39):
    if i==0:
        train_data = class_data[i][0]
        val_data = class_data[i][1]
    else:
        train_data = torch.cat((train_data, class_data[i][0]),dim=0)
        val_data = torch.cat((val_data, class_data[i][1]),dim=0)
    
train_label = [i for i in range(39) for _ in range(80)]
val_label = [i for i in range(39) for _ in range(20)]

train_dataset = GestureDataset(train_data, train_label)
val_dataset = GestureDataset(val_data, val_label)


# form train & val dataloader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) # enable shuffle and drop_last
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# save train % val dataloader
torch.save(train_loader, '/mnt/fyp/data/train_loader.pt')
torch.save(val_loader, '/mnt/fyp/data/val_loader.pt')