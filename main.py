
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data_test = pd.read_csv('/Users/max/PycharmProjects/NN From Scratch/venv/mnist_test.csv')
data_train = pd.read_csv('/Users/max/PycharmProjects/NN From Scratch/venv/mnist_train.csv')

m1, n1 = data_test.shape
m2, n2 = data_train.shape
data_test = data_test.T
data_train = data_train.T


test_label = data_test[0]
test_values = data_test[1:n1]
test_values = test_values / 255
_, m_test = test_values.shape

train_label = data_train[0]
train_values = data_train[1:n2]
train_values = train_values / 255
_, m_train = train_values.shape

print(train_label)

print("train:", m_train, "test", m_test)


