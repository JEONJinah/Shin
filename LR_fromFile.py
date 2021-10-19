import torch
import torch.nn as nn # nn.linear 라이브러리를 사용하기 위해 import
# F.mse(mean squared error) <- linear regression, LOSS Function 존재
# Classification problem에서 사용하는 loss function : Cross-Entropy
import torch.nn.functional as F
import torch.optim as optim # SGD, Adam, etc.최적화 라이브러리

# import file_read
from file_read import read_file

house_info, cnt = read_file("sell_house.txt")

