# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:31:12 2024
@author: Fuzzy logic lab
"""
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split

import data_managment
# %% Data extraction
training_generator = data_managment.training_generator
test_generator = data_managment.test_generator
# %% Device selection
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# %% Model description
network = nn.Sequential(
            
            nn.Conv2d(2, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,4))

network.to(device)

# %% Loss criterion
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.003, momentum=0.9)

epochs = 10000
batch = 40

for epoch in range(epochs):
    # Training
    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.float().to(device), local_labels.to(device)
        # Optimization process
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = network(local_batch)
        loss = criterion(outputs, local_labels)
        loss.backward()
        optimizer.step()
        loss_train = loss.item()
        print("Train loss: {} Epoch:{}".format(loss_train,epoch))
    if loss_train <= 0.01:
        break

len(training_generator)
    
# %% Accuracy computing
def compute_acc(generator):
    acc = 0
    lenght = 0
    # Iterate over the dataset
    for local_batch, local_labels in generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.float().to(device), local_labels.to(device)
        # Evaluate the model
        pred = network(local_batch)
        # Convert to numpy for statics
        pred = pred.cpu().detach().numpy()
        pred = np.argmax(pred, axis= 1)
        y = local_labels.cpu().detach().numpy()
        y = np.argmax(y, axis= 1)
        acc = acc + np.count_nonzero(pred==y)
        lenght = lenght + len(local_batch)
    print(acc/lenght)

# %% Compute train and test accuracy
compute_acc(training_generator)

compute_acc(test_generator)
