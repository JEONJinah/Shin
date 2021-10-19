import torch
from Own_ML_Class import LogisticRegression
import torch.nn.functional as F
import torch.optim as optim

import numpy as np   # for reading the csv files

# 1. Data 를 준비(read data --> extract the training data to the entire datasets)
# ==> Test Dataset

data = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

# prepared training dataset
train_x = data[0:-20, 0:-1]  # Features
train_y = data[0:-20, [-1]]

# testset 분리
test_x = data[-20:, 0:-1]
test_y = data[-20:, [-1]]

# pytorch 로 학습할 경우, np 에서 tensor 타입으로 변경해야 함
x_train = torch.FloatTensor(train_x)
y_train = torch.FloatTensor(train_y)

# 모델을 정의하고, 어떤 최적화 함수를 사용할지에 대해서 정의

model = LogisticRegression(8, 1)

optimizer = optim.SGD(model.parameters(), lr=1.0)

#=========================학습을 위해 셋팅 완료

np_epochs = 2000
for epoch in range(np_epochs + 1):
    hx = model.forward(x_train) # model.forward(x_train)
    cost = F.binary_cross_entropy(hx, y_train)

    optimizer.zero_grad()
    cost.backward()  # 미분, 기울기 계산
    optimizer.step()  # 최적화 (SGD 수행 및 w, b update)

    if epoch % 100 == 0:
        prediction = model(torch.FloatTensor(test_x)) >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == torch.FloatTensor(test_y)
        accuracy = correct_prediction.sum().item() / len(correct_prediction)
        print(f'Epoch: {epoch}/{np_epochs} Cost: {cost.item()} Accuracy: {accuracy * 100}%')



