# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:31:12 2024
@author: Fuzzy logic lab
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import pandas as pd

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
loss_vector_train = []
loss_vector_validation = []

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
    # Validation
    for local_batch, local_labels in test_generator:
        # Transfer to GPU
        val_batch, val_labels = local_batch.float().to(device), local_labels.to(device)
        out = network(val_batch)
        loss_val = criterion(out, val_labels)
        loss_validation = loss_val.item()
        print("Validation loss: {} Epoch:{}".format(loss_validation,epoch))
    loss_vector_train = np.append(loss_vector_train, loss_train)
    loss_vector_validation = np.append(loss_vector_validation, loss_validation)
    if loss_train <= 0.01:
        break

len(training_generator)
    
# %% Accuracy computing
def compute_acc(generator):
    acc = 0
    lenght = 0
    pred_vector = []
    targ_vector = []
    # Iterate over the dataset
    for local_batch, local_labels in generator:
        # Transfer to device
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
        pred_vector = np.append(pred_vector, pred)
        targ_vector = np.append(targ_vector, y)
    print(acc/lenght)
    return pred_vector, targ_vector

# %% Compute train and test accuracy
train_p, train_t = compute_acc(training_generator)
test_p, test_t = compute_acc(test_generator)
# %% Plot train and validation losses
plt.plot(loss_vector_train, label='Train loss')
plt.plot(loss_vector_validation, label='Validation loss')

plt.xlabel('Epochs')
plt.ylabel('Cross entropy loss')
plt.title('Training and validation losses (general)')
plt.legend()

plt.show()
# %% Confussion matrix plot
from sklearn.metrics import ConfusionMatrixDisplay

df_true = pd.DataFrame({"y_true": test_t})
df_pred = pd.DataFrame({"y_pred": test_p})

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(df_true, df_pred, ax=ax, display_labels=["Healthy","Inner race","Ball","Outer race"],cmap=plt.cm.Blues)
_ = ax.set_title("Test confusion matrix (general)")


df_true = pd.DataFrame({"y_true": train_t})
df_pred = pd.DataFrame({"y_pred": train_p})

fig, ax = plt.subplots(figsize=(7, 5))

ConfusionMatrixDisplay.from_predictions(df_true, df_pred, ax=ax, display_labels=["Healthy","Inner race","Ball","Outer race"],cmap=plt.cm.Blues)
_ = ax.set_title("Train confusion matrix (general)")
