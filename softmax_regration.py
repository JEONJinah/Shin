import torch
import torch.nn.functional as F
import torch.optim as optim

# F.softmax
z = torch.FloatTensor([1, 2, 3])
# exp(1) / exp(1) + exp(2) + exp(3)
hypothesis = ( torch.exp(z[0]) / (torch.exp(z[0])+torch.exp(z[1])+torch.exp(z[2])),
               torch.exp(z[1]) / (torch.exp(z[0])+torch.exp(z[1])+torch.exp(z[2])),
               torch.exp(z[2]) / (torch.exp(z[0])+torch.exp(z[1])+torch.exp(z[2])) )

softmax_hypothesis = F.softmax(z, dim=0)
print(hypothesis, softmax_hypothesis)

# x_train의 각 샘플은 4개 특징을 가지고 있고 총 8개 샘플이 존재
x_train = [[1, 2, 1, 1], # ==> 2 [0, 0, 1]
           [2, 1, 3, 2], # [0, 0, 1]
           [3, 1, 3, 4], # [0, 0, 1]
           [4, 1, 5, 5], # [0, 1, 0]
           [1, 7, 5, 5],
           [1, 2, 5, 6],
           [1, 6, 6, 6],
           [1, 7, 7, 7]]

# Softmax, Multinomial Classification (n개 중에 한개 선택, 확률값이 가장 큰 class 하나 선택)
# Number of the Class : 3개 (0, 1, 2)
y_train = [2, 2, 2, 1, 1, 1, 0, 0]

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

# one-hot encoding tensor 생성
# 값이 0인 8x3 2D Tensor 생성

# scatter(흩어짐), unsqueeze(1) ==> 8x1, unsqueeze(0) ==> 1x8
# scatter_ <- 계산된 결과 그대로 y_one_hot 변수에 assign
y_one_hot = torch.zeros(8, 3)
y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)

# 8x4 * W = 8x3
# model 초기화
W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# optimizer 설정
optimizer = optim.SGD([W, b], lr=0.1)

nb_epochs = 1000
for epoch in range(nb_epochs + 1):
    # hypothesis
    hypothesis = F.softmax(x_train.matmul(W)+ b, dim=1)

    # cost, loss 함수 정의
    # 1/n(target(GT) * -log(h(x)) : h(x) ==> hypothesis
    # *: elements-wise multiplication (a.mul(b))
    cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{nb_epochs} Cost: {cost.item()}')