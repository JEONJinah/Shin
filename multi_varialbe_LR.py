import torch
import torch.nn as nn # nn.linear 라이브러리를 사용하기 위해 import
# F.mse(mean squared error) <- linear regression, LOSS Function 존재
# Classification problem에서 사용하는 loss function : Cross-Entropy
import torch.nn.functional as F
import torch.optim as optim # SGD, Adam, etc.최적화 라이브러리

# 임의 데이터 생성
# 입력이 1, 출력이 1
# Multi-variable linear regression (입력 3, 출력 1)
# input(x_train) 4x3 2D Tensor 생성
x_train = torch.FloatTensor([[90, 73, 89],
                             [66, 92, 83],
                             [86, 87, 78],
                             [85, 96, 75]])

# y_train (GT)
y_train = torch.FloatTensor([[152],
                             [185],
                             [100],
                             [193]])

# 모델 선언 및 초기화
# y = WX (w1*x1 + w2*x2...wn*xn + b)
# nn.Linear(input_dim, output_dim)
# 초기화
# w = randn(1)
# model.paramters (weight: 3, bias: 1)
# weight, bias : 랜덤한 값으로 자동 셋팅
model = nn.Linear(3, 1) # get_weights()함수 참고..
# model.parameters() 최적화, w,b로 미분을 해야하므로 (requires_grad=True) 셋팅된 것을 확인할 수 있음.
print(list(model.parameters()))

optimizer = optim.SGD(model.parameters(), lr=0.01)  # learning_rate 설정: 노가다하면서.. 구하세요.

# iteration 횟수 지정 (epoch 횟수 지정)
# epoch : 전체 훈련 데이터에 대해 경사 하강법을 적용하는 횟수 (2000번을 돌면서 w, b 값을 update)
nb_epochs = 2000
for epoch in range(nb_epochs+1):
    # H(x) 계산 wx+b를 한번 계산한 결과값을 pred 변수에 assign
    # x_train = 입력 데이터 (1, 2, 3), w (0.6242), b (-0.1192)
    # 추정값 = w*x_train+b
    pred = model(x_train)

    # cost 계산 (loss function : Mean Square Error)
    # Cost fuction, loss Function --> Cost, Loss, Error
    # mse = mean(sum(pow(y, y^))))
    cost = F.mse_loss(pred, y_train) # y_train (GT, 결과, 2, 4, 6)

    # SGD를 이용해서 최적값 도출하는 부분 (w,b 값을 조정)
    optimizer.zero_grad() # gradient 계산 시 zero 초기화가 들어가 있지 않으면 누적된 값으로 적용
    cost.backward() # 실제 기울기 값 계산하는 부분
    optimizer.step() # w, b 값을 update 하는 부분

    # 100번 마다 로그 출력
    if epoch % 100 == 0:
        tmp = list(model.parameters())
        print(f'Epoch: {epoch:4d} Cost : {cost.item(): .6f}')

print(f'w, b: {tmp[0]}, {tmp[1]}')

new_var = torch.FloatTensor([[73, 80, 75]])
# 152에 근접한 값이 출력이 되면 학습이 잘 된 것으로 판단.
pred_y = model(new_var)   # model.forward(new_var)
