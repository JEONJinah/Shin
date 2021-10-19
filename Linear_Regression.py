import torch
import numpy as np
import matplotlib.pyplot as plt

# function name : get_data
# param in : None
# param out : will be returned trained x and y
# Description
"""
데이터를 가지고 오는 함수
np.array를 tensor로 변경한 후 Tensor 값을 반환
"""
def get_data():
    train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1], dtype='f')
    train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3], dtype='f')

    # Hypothesis 함수 구현 시 matrix multiplication으로 구현
    # train_x의 타입을 벡터에서 매트릭스로 변경
    # current train_x.shape ([17,]) ==> [17x1]
    # view 대신 unsqueeze(1) 사용 가능 (벡터를 matrix)
    x = torch.from_numpy(train_X).view(17, 1)
    y = torch.from_numpy(train_Y)

    return x, y

# function name : get_weights
# param input : None
# param output : random weightand bias values
# Description
"""
- weights, bias value를 random_normal한 값으로 셋팅
- randn (random normal) : standard normal distribution
(평균이 0, 표준편차 1인 가우시안 정규분포: 표준 정규분포)
- -1~0~1 : 0.34%, 0.34% (about 68% 존재)
- -2~-1, 1~2: 0.136,.. (about 27%)
==> randn : 평균에 가까운 (0에서부터 편차가 1인 부분의 데이터가 가장 많음) : 음수 출력 가능
- rand : 0~1 사이의 값을 균등하게 생성 (음수값 x)
- required_grad True setting : w, b로 미분을 수행하기 때문에
- w, b 값이 iteration 마다 미분되어 최적화 되기 때문에 grad=True로 셋팅 
"""

def get_weights():
    w = torch.randn(1) # 1: size(output size)
    w.requires_grad = True
    b = torch.randn(1)  # 1: size(output size)
    b.requires_grad = True

    return w, b

# function name : simple_network (or hypothesis_func)
# H(X) = WX + b를 구하기 위한 함수
# params in : train_data x (입력 데이터 = 학습 데이터)
# params out : 가설 함수의 출력값 (y = weight * x + b)
# Description
"""
예측 값을 구하기 위해서 matrix multiplication 수행
H(x) = Wx + b
x = 17x1 w = 1x1 b = 1x1
* notation : matrix multiplication
17x1 * 1x1 = 17x1 + b (1x1) ==> tf, pytorch --> 가능하게 해주는 것 : Broadcasting 방법
simple network 함수에서 출력되는 것은 예측값(y^)
"""
def simple_network(x):
    # 출력 값 : H(x) = w * x (matrix multiplication) + b
    y_pred = torch.matmul(x, w) + b

    return y_pred

# function name : loss_fn (or cost_function)....
# params in: y(GT 값), y_pred or y_hat 값을 입력
# params out: <loss, Error, cost> 값을 반환
# Description
"""
Loss function : MSE (Mean Squared Error)
"""

def loss_fn(y, y_pred):
    loss = torch.mean((y-y_pred).pow(2).sum()) # loss function 정의
    # sum_0 = (y[i] - y_pred[i])**2
    # sum_0 += (y[i] - y_pred[i])**2
    # sum_0 = sum_0/17
    for param in [w, b]:
        if not param.grad is None: param.grad.data.zero_()
    loss.backward() # MSE 기울기 계산 (LOSS function, MSE가 w, b의 함수로 정의되어 있으므로 w, b에 따른 수행)
    return loss.data

# function name : optimize
# params in : lr (learning rate)
# params out : None
# Description
"""
W, b 값을 업데이트 하면서 최적화 하는 부분
매 iteration 마다 w, b 값을 업데이트 (함수 내에서는 한 번만 update)
"""
def optimize(learning_rate):
    w.data -= learning_rate * w.grad.data
    b.data -= learning_rate * b.grad.data


def plot_variable(x, y, z='', **kwargs):
    l = []
    for a in [x, y]:
        l.append(a.data)
    plt.plot(l[0], l[1], z, **kwargs)

if __name__ == '__main__':

    # # 자동미분(autograd) ==> grad.data.zero_()
    # w = torch.tensor(2.0, requires_grad=True)
    # nb_epochs = 10
    # for epoch in range(nb_epochs):
    #     if not w.grad is None: w.grad.data.zero_()
    #     z = 2*w #==> 2
    #     z.backward() # dz/dw (z 함수를 w로 미분)
    #
    #     print(f'함수 z를 w로 미분한 값 : {w.grad}')


    # 학습할 데이터의 분포를 확인
    # Plot을 통해 그림으로 확인
    # train_x : 학습할 데이터, train_y : train_x에 대응하는 Ground Truth 값
    train_x, train_y = get_data()
    plot_variable(train_x, train_y, 'ro')
    plt.show()

    # Weight, Bias 값을 랜덤한 값으로 셋팅하기 위해 get_weight 함수 호출
    w, b = get_weights() # w, b : 학습 파라미터

    learning_rate = 1e-4

    for i in range(500):
        y_pred = simple_network(train_x)  # wx + b 를 계산하는 함수 (forward 함수)
        loss = loss_fn(train_y, y_pred)  # w,b 로 Loss(MSE)함수를 미분하고, 미분된 값 저장 (기울기 계산)

        if i % 50 == 0:
            print(f'epoch: {i}, Loss: {loss}, weight: {w}, Bias: {b}')
        optimize(learning_rate)  # 오차가 최소화되도록 w, b 값을 업데이트 하는 부분

    plot_variable(train_x, train_y, 'ro')
    plot_variable(train_x, y_pred, label='Fitted line')
    plt.show()