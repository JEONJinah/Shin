import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

house_info, cnt = read_file("sell_house.txt")
# print(house_info)

A = np.array(house_info)

x_train = A[:-5, 1:-1]
y_train = A[:-5, -1:]

x_test = A[-5:, 1:-1]
y_test = A[-5:, -1:]

