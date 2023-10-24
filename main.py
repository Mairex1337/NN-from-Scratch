import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt

# ____________________________________________________________________________________________________________________
# Input parsing

data_test = pd.read_csv('/Users/max/PycharmProjects/NN From Scratch/venv/mnist_test.csv')
data_train = pd.read_csv('/Users/max/PycharmProjects/NN From Scratch/venv/mnist_train.csv')
# Turn data into Tensors
# Transpose the matrices
# Separate out labels
test_tensor = torch.tensor(data_test.values.T[1:])
test_label = torch.tensor(data_test.values.T[0:1])
train_tensor = torch.tensor(data_train.values.T[1:])
train_label = torch.tensor(data_train.values.T[0:1])

# Normalize values to be between 0 and 1
test_tensor = test_tensor / 255
train_tensor = train_tensor / 255

# Debug
#print(test_tensor.shape)
#print(train_tensor.shape)
#print(test_label.shape)
#print(train_label.shape)

# ____________________________________________________________________________________________________________________

# Initializing the network

# weights of input
w1 = 2 * torch.rand(100, 784) - 1
# biases of the hidden layer
b1 = torch.zeros(100)
# weights of hidden layer
w2 = 2 * torch.rand(100, 10) - 1
# biases of output
b2 = torch.zeros(10)
print(b2)

# ____________________________________________________________________________________________________________________

# forward pass

def forward_pass(input_data, w1, b1, w2, b2):


