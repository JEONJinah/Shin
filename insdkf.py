import torch
import numpy as np
import matplotlib.pyplot as plt

# function name : get_data
# param in: None
# param out: will be returned, trained x and y
#  Description
"""
데이터를 가지고 오는 함수
np.array를 tenor로 변경한 후 Tensor 값을 반환
"""

def get_data():
    train_X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])

    train_Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

    # Hypothesis 함수 구현 시 matrix multiplication으로 구현
    # train_X의 타입을 벡터에서 메트릭스로 변경.
    # current train_x.shape([17,]) ==> [17x1]
    # 가지고 있는 원소의 갯수는 늘 변하지 않는다는 걸 유념해야 한다.

    x = torch.from_numpy(train_X).view(17, 1)
    # view 대신 사용가능한 것(벡터를 matrix) --> unsqueeze(1)
    y = torch.from_numpy(train_Y)


    return x, y

# *args ==> 여러 개의 가변 인자를 받을 때 사용한다.
# **kwargs ==> key와 value값을 넘길 때 사용하는 parameter
def plot_variable(x, y, z='', **kwargs):
    l = []

    for a in [x, y]:
        l.append(a.data)

    plt.plot(l[0], l[1], z, **kwargs)


if __name__ == '__main__':

    #Data 분포를 확인(학습할 데이터의 분포를 확인)
    # Plot을 통해 그림으로 확인
    # train_x : 학습할 데이터, train_y : train_x에 대응하는 GT(Ground Truth)값이 전달 되어짐.
    train_x, train_y = get_data()
    plot_variable(train_x, train_y, 'ro')
    plt.show()