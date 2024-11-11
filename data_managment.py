# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:21:00 2024

@author: Fuzzy Logic
"""
import torch
import torchaudio
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision
from sklearn.model_selection import train_test_split

# %% Reading dataset
# Handle dataset
data_1 = pd.read_csv(r'\4C_general_007.csv', header = None)
data_2 = pd.read_csv(r'\4C_general_014.csv', header = None)
data_3 = pd.read_csv(r'\4C_general_021.csv', header = None)

data = pd.concat([data_1, data_2, data_3], ignore_index=True, axis=0)

# %% Structural variables
size = 16*16
n_sample = int(len(data)/size)

#%%  Normal dataset
n_data = data.values[0:n_sample*size,[0,4]]
n_data = np.reshape(n_data, (n_sample,2,16,16))
n_label = np.repeat(0, n_sample)

# Inner race fault dataset
ir_data = data.values[0:n_sample*size,[1,5]]
ir_data = np.reshape(ir_data, (n_sample,2,16,16))
ir_label = np.repeat(1, n_sample)

# Ball fault dataset
b_data = data.values[0:n_sample*size,[2,6]]
b_data = np.reshape(b_data, (n_sample,2,16,16))
b_label = np.repeat(2, n_sample)

# Outer race fault dataset
or_data = data.values[0:n_sample*size,[3,7]]
or_data = np.reshape(or_data, (n_sample,2,16,16))
or_label = np.repeat(3, n_sample)

# %% Organizing the dataset
x_data = np.concatenate((n_data,ir_data,b_data,or_data))
x_label = np.concatenate((n_label, ir_label, b_label, or_label), axis=0)
# train and test split
x_train, x_test, y_train, y_test = train_test_split(x_data, x_label, test_size=0.2)

# %% Tesnor transformations
x_train = torch.tensor(x_train).float()
y_train = torch.tensor(y_train).long()

x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).long()
# One hot codification
y_train = F.one_hot(y_train, num_classes=4).float()
y_test = F.one_hot(y_test, num_classes=4).float()

# %% Dataloader class creation
class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, data, labels):
        'Initialization'
        self.labels = labels
        self.data = data

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

  def __getitem__(self, index):
        'Generates one sample of data'
        X = self.data[index]
        y = self.labels[index]

        return X, y
    
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

# Parameters
params = {'batch_size': 500,
          'shuffle': True,
          'num_workers': 0}

# Generators
training_set = Dataset(x_train, y_train)
training_generator = torch.utils.data.DataLoader(training_set, **params)

test_set = Dataset(x_test, y_test)
test_generator = torch.utils.data.DataLoader(test_set, **params)
