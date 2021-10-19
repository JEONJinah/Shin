import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 입력이 1, 2, 3
# 출력이 2, 4, 6
# 나오는 weight 값과 bias 값을 구함

# 2D-Tensor 생성 및 값을 셋팅
# 3x1 matrix 2개 ==> 2D-Tensor
x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])
print(x_train.shape, y_train.shape)

# 모델을 선언하는 부분
# 사용하고자 하는 모델은 linear regression, input_dim = 1(feature 가 한개), output_dim = 1
# Feature 수가 늘어날 경우에 nn.Linear(n, 1)로 셋팅
# 최종적으로 모델에서 출력되는 값은 weight, bias 값이며,
# 학습을 시키기 위해서는 해당 값을 초기화 해야하고, 랜덤 값을 지정해야 함.
# nn.Linear 를 사용할 경우, 라이브러리에서 w, b 값을 랜덤한 값으로 생성해 줌
model = nn.Linear(1, 1)
print(list(model.parameters()))

# optimizer 설정. 확률기반 경사 하강법(SGD) 사용하고, learning_rate 지정
# lr을 셋팅하는 가이드는 없음
# 따라서 cost, loss, error 값의 변화를 보면서 셋팅을 휴리스틱하게 지정해야 함.
optimizer = optim.SGD(model.parameters(), lr=0.01)

# iteration  횟수 지정(epoch 횟수 지정)
# epoch:  전체 훈련 데이터에 대해 경사 하강법을 적용하는 횟수 (2000번을 돌면서 w, b 값을 update)
nb_epochs = 2000
for epoch in range(nb_epochs):
    # H(x) 계산 wx+b를 한번 계산한 결과 값을 pred 변수에 assign
    # x_train = 입력 데이터 (1, 2, 3), x(-0.3751), b(0.8476)
    # 추정값 = w*x_train+b
    pred = model(x_train)

    # cost 계산 (Loss function: Mean Square Error)
    # Cost function, Loss Function, --> Cost, Loss, Error
    # mse = mean(sum((pow(y - y^))))
    cost = F.mse_loss(pred, y_train)  # y_train (GT, 결과, 2, 4, 6)

    # SGD 를 이용해서 최적값 도출하는 부분(w, b 값을 조정)
    optimizer.zero_grad() # gradient 계산을 할건데, zero 초기화가 들어가있지 않으면 누적된 값으로 적용
    cost.backward() # 실제 기울기 값 계산하는 부분
    optimizer.step() # w, b 값을 update 하는 부분

    # 100 번마다 로그 출력
    if epoch % 100 == 0:
        tmp = list(model.parameters())
        print(f'Epoch: {epoch:4d} Cost: {cost.item(): .6f}')

print(f'W, b: {tmp[0]}, {tmp[1]}')
