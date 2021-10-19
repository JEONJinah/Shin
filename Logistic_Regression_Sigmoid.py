import torch
import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# function name: get_data
# param in: None
# param out: will be returned trained x and y
# Descrpition
"""
데이터를 가지고 오는 함수
np.array를 Tensor로 변경한 후 Tensor 값을 반환
"""
def get_data():
    train_X = np.array([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                        7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
    train_Y = np.array([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                        2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])

    # Hypothesis 함수 구현 시 matrix multiplication으로 구현
    # train_X의 타입을 vector에서 matrix로 변경
    # current train_X.shape([17,]) -> [17x1]
    # view대신 사용 가능한 것(벡터를 매트릭스로) = unsqueeze(1)
    x = torch.from_numpy(train_X)
    y = torch.from_numpy(train_Y)
    return x, y

#function name: get_weights
# params input: None
# params output: random weight and bias values
# Description
"""
- weights, bias values를 random_normal한 값으로 셋팅
- randn(random normal): standard normal distribution(평균이 0, 표준편차가 1인 가우시안 정규분포: 표준 정규분포)
- -1~0~1: 0.34%, 0.34% 사이에 대충 68%가 존재
- -2~-1, 1~2 사이에 0.136,... 대략 27%가 존재
- randn: 평균에 가까운(0에서부터 편차가 1인) 부분의 데이터가 가장 많음, 음수 출력 가능
- vs rand: 0~1 사이의 값을 균등하게 생성, 음수 값이 나올 수 없음
- required_grad=True로 셋팅: w, b로 미분을 수행하기 때문에
- w, b 값이 iteration마다 미분되어 최적화되어지기 때문에, grad=True 로 셋팅
"""
def get_weights():
    w = torch.randn(1) # 1:size(output size)
    w.requires_grad = True
    b = torch.randn(1)
    b.requires_grad = True
    return w, b


# function name: simple_network(or hypothesis_func)
# H(X) = WX + b를 구하기 위한 함수
# params in: train_data x(입력데이터=학습데이터)
# params out: 가설함수의 출력값 y(y = weight * x + b
# Description
"""
예측 값을 구하기 위해서 matrix multiplication 수행
H(x) = Wx + b
x = 17x1, W = 1x1, b = 1x1
*notation: matrix multiplication
17x1 * 1x1 => 17x1 + b(1x1) ==> pytorch의 Broadcasting 방법으로 가능하게 해줌
simple network 함수에서 출력되는 것은 예측값(y^)
"""
def simple_network(x):
    # 출력값: H(x) = w * x(matrix multiplication) + b
    y_pred = torch.matmul(x.float(), w.float()) + b
    return y_pred


# function name: loss_fn
# params in: y(GT값), y_pred or y_hat 값을 입력
# params out: <Loss, Error, cost> 값을 반환
# Description
"""
Loss function: MSE(Mean Squared Error)
"""
def loss_fn(y, y_pred):
    loss = torch.mean((y_pred - y).pow(2).sum())
    # sum_0 = (y[i] - y_pred[i]) ** 2
    # sum_0 += (y[i] - y_pred[i])**2
    # sum_0 = sum_0 / 17
    for param in [w, b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward()  # Backward를 호출하면 해당 수식(MSE)의 기울기를 계산
    return loss.data


# function name: optimize
# params in: lr(learning_rate)
# params out: None
# Description
"""
W, b 값을 업데이트 하면서 최적화 하는 부분
매 iteration마다 w, b 값을 업데이트
"""
def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


# 데이터가 어떻게 생겼는지 보기위해
def plot_variable(x, y, z='', **kwargs):
    l = []
    for a in [x, y]:
        l.append(a.data)
    plt.plot(l[0], l[1], z, **kwargs)
    plt.plot([0, 0], [1, 0], ":")
    plt.title("Sigmoid Function")


if __name__ == "__main__":


    # Data 분포를 확인 (학습할 데이터의 분포를 확인)
    # plot을 통해 그림으로 확인
    # train_x: 학습할 데이터, train_y: train_x에 대응하는 GT(Ground Truth)값이 전달 되어짐
    train_x, train_y = get_data()
    plot_variable(train_x, train_y, 'ro')
    plt.show()

    # 파이토치의 자동미분(autograd) --> grad.data.zero_() 함수가 필요한 이유
    w = torch.tensor(2.0, requires_grad=True)
    nb_epochs = 10
    for epoch in range(nb_epochs):
        z = 2 * w  # --> 2가 나옴
        z.backward()  # dz/dw(z함수를 w로 미분)
        print(f'함수 z를 w로 미분한 값: {w.grad}')

    for epoch in range(nb_epochs):
        w.grad.data.zero_()
        z = 2 * w
        z.backward()
        print(f"함수 z를 w로 미분한 값: {w.grad}")


    # weight와 bias 값을 랜덤한 값으로 셋팅하기 위해 get_weights 함수 호출
    w, b = get_weights(train_x)  # w, b: 학습 파라미터

    learning_rate = 1e-4
    for i in range(500):
        y_pred = simple_network(train_x)  # Wx+b를 계산하는 함수(forward 함수)
        loss = loss_fn(train_y, y_pred)

        if i % 50 == 0:
            print(f'epoch:{i}, loss{loss}, weights:{w}, bias:{b}')
        optimize(learning_rate)

    plot_variable(train_x, train_y, 'ro')
    plot_variable(train_x, y_pred, label='Fitted Line')
    plt.show()

    # b 를  0.5, 1, 1.5로 해서 그래프 그려봅시다.

