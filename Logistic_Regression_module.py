import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# data 준비
x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]


# list to tensor for training (using torch)
x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

"""
모델 아키텍처 정의
nn.Sequential : layer를 차례로 쌓을 수 있도록 하는 것
wx+b(linear regression) + sigmoid를 연결하기 위해 nn.Sequential 사용
"""

model = nn.Sequential (
    # nn.Conv2d(3, ..)
    # nn.Flatten()
    nn.Linear(2, 1), # Wx + b (input dim, output dim)
    nn.Sigmoid()
)


# 최적화 함수 설정
optimizer = optim.SGD(model.parameters(), lr=1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    """
    1. 가설 함수 정의 및 계산 (forward)
    2. cost function 정의 및 계산
    3. w, b값 업데이트 (using optimizer(미분)): backward
    """
    # hypothesis function : sigmoid
    hypothesis = model(x_train)

    # binary cross entropy library
    cost = F.binary_cross_entropy(hypothesis, y_train)


    # cost 값을 최적화 (minimize) 하고 w, b값 업데이트
    optimizer.zero_grad()
    cost.backward() # Loss(cost) function 미분 (기울기 계산)
    optimizer.step() # lr만큼 내려가면서 w, b값 업데이트

    if epoch % 100 == 0:
        # 예측 값이 0.5를 넘으면 true
        prediction = hypothesis >= torch.FloatTensor([0.5])
        print(prediction)
        correct_prediction = prediction.float() == y_train
        print(correct_prediction)
        # accuracy: correct_prediction.sum() / 전체갯수(6)
        # correct_prediction.sum() ==> print(tensor(4) or tensor(5) ... etc)
        # ==> 4 또는 5를 가지고 오기 위해서 .item()을 사용
        accuracy = correct_prediction.sum().item()/len(correct_prediction)
        print(f'Epoch {epoch}/{nb_epochs} Cost: {cost.item()} Accuracy{accuracy * 100}%')
