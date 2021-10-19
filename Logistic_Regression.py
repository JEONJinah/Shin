import torch
# import torch.nn as nn # nn.Linear() 사용 시 필요
# import torch.nn.functional # Loss function 계산 시 필요
import torch.optim as optim
import numpy as np

# data 준비
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]


# list to tensor for training (using torch)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

# 가설 함수 (1/1+exp^-(wx+b))
# 6x2 * 2x1 = 6x1 ==> +b
# Logistic Regression hypothesis 함수는 sigmoid이기 때문에 0으로 셋팅 시 초반 출력값 0.5
W = torch.zeros((2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 최적화 함수 설정
optimizer = optim.SGD([W, b], lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    """
    1. 가설 함수 정의 및 계산 (forward)
    2. cost function 정의 및 계산
    3. w, b값 업데이트 (using optimizer(미분)): backward
    """
    # hypothesis function : sigmoid
    hypothesis = 1 / (1 + torch.exp(-(x_train.matmul(W)+b)))
    # hypothesis = torch.sigmoid(x_train.matmul(W)+b)

    # cost function 정의 (Log 함수 이용)
    # binary classification 특성 상 0~1 사이 값이 출력되어야 하고, -log(x){0<=x<=1}의 형태가
    # 구하고자 하는 loss에 적합해야 함
    cost = -(y_train * torch.log(hypothesis) + ((1 - y_train) * torch.log(1 - hypothesis))).mean()

    # cost 값을 최적화 (minimize) 하고 w, b값 업데이트
    optimizer.zero_grad()
    cost.backward() # Loss(cost) function 미분 (기울기 계산)
    optimizer.step() # lr만큼 내려가면서 w, b값 업데이트

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{nb_epochs} Cost: {cost.item()}')

print(W, b)

# 학습이 잘 되었으면
# 3개는 0에 가까운 값이 출력, 3개는 1에 가까운 값 출력
hypothesis = torch.sigmoid(x_train.matmul(W) + b)
print(hypothesis)

data = np.loadtxt('./data-03-diabetes.csv', delimiter=',', dtype=np.float32)
print(data)

pred = []
for i in list(hypothesis):
    if i >= 0.5: pred.append(1)
    else: pred.append(0)
print(pred)

# [value for i in list(hypothesis) if ...]
# for문과 if문 없이 작성할 수 있나요?
predition = hypothesis >= torch.FloatTensor([0.5])
print(predition)  # [F,F,F,T,T,T]