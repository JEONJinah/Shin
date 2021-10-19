# projection vector visualization
# b 벡터를 a 벡터로 projection
# numpy : array, vector, matrix 표현
# dot product, outer product, matrix multiplication 쉽게 수행하기 위해 사용
import numpy as np
import plotly
import plotly.graph_objs as go

# b 벡터를 a 벡터로 projection visualization 해 주는 함수 구현
# 입력 매개 변수 : 2개 (벡터가 두개 있어야 하므로)
def proj_to_line(vector_a, vector_b):

    # vector a를 3개로 분리
    # plot 하기 위해 분리 진행
    # a = [1, 2, 2]
    # a1 = 1, a2 = 2, a3 = 2
    # x_axis = [0, a1]
    # y_axis = [0, a2]
    # z_axis = [0, a3]
    a1, a2, a3 = vector_a
    b1, b2, b3 = vector_b

    # 함수로 입력되는 인자는 vertor가 아닌 리스트로만 전달 가능
    # numpy 이용해서 vector로 변경 후 넘겨도 됨
    # 함수 내에서는 리스트를 vector로 변경해서 사용 (numpy의 array로 변환)
    vect_a = np.array(vector_a)
    vect_b = np.array(vector_b)

    # projection matrix : outer production / inner production(dot production)
    # V*V.T(외적) / V.T*V(내적) ==> projection matrix
    # VV.T (두 개의 vector를 외적하면 matrix가 생성 : 3x1 * 1x3 ==> 3x3)
    # V.TV (두 개의 vector를 내적하면 실수(real number) 생성 : 1x3 * 3x1 = 1x1 (R))
    P_a = np.outer(vect_a, vect_a) / vect_a.dot(vect_a)

    # projection vector = projection matrix * b.T
    # projection vector = 3x1 (projection matrix 3x3 * 3*1 = 3x1)
    # p1, p2, p3로 분리해서 저장 (그리기 위해)
    p1, p2, p3 = P_a.dot(vect_b.T)

    data = []
    vector = go.Scatter3d(x=[0, a1], y=[0, a2], z=[0, a3], marker=dict(size=[0,5], color=['pink'],
                                                                       line=dict(width=5, color='gray')),
                                                                        name='a')

    data.append(vector)

    vector = go.Scatter3d(x=[0, b1], y=[0, b2], z=[0, b3], marker=dict(size=[0, 5], color=['skyblue'],
                                                                       line=dict(width=5, color='gray')),
                          name='b')
    data.append(vector)

    vector = go.Scatter3d(x=[0, p1], y=[0, p2], z=[0, p3], marker=dict(size=[0, 5], color=['green'],
                                                                       line=dict(width=5, color='gray')),
                          name='projection')
    data.append(vector)

    vector = go.Scatter3d(x=[b1, p1], y=[b2, p2], z=[b3, p3], marker=dict(size=[0, 5], color=['violet'],
                                                                       line=dict(width=5, color='gray')),
                          name='error')
    data.append(vector)

    fig = go.Figure(data=data)
    fig.show()





proj_to_line([1, 2, 2], [1, 1, 1])