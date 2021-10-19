import torch

# torchvision.dataset : 배포되어 있는 open dataset 다운로드 가능
# torchvision: 이미 잘 알려진 모델 다운로드 및 실행이 가능, 이미지 전처리 등 여러 패키지들이 포함함
import torchvision.datasets as dsets
import torchvision.transforms as transforms # 데이터들을 텐서로 일괄 변경
from torch.utils.data import DataLoader # DataLoader를 사용해서 shuffle, batch size 등을 셋팅하기 위해
import torch.nn as nn
import matplotlib.pyplot as nn
import matplotlib.pyplot as plt
import random

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

# Set hyperparameters
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(root='MNIST_data/',
#  실행마다 다운 받는 것이 아니라, 해당 폴더에 데이터가 없을 경우, 자동으로 데이터 셋 다운로드 하기 위해
                          train=False, transform=transforms.ToTensor(), download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False, transform=transforms.ToTensor(), download=True)
data_loader = DataLoader(dataset=mnist_train,
                         batch_size=batch_size, shuffle=True, drop_last=True)
# mnist = 28x28x1, 0~9까지의 숫자 중 하나를 인식하기 위해 사용하는 데이터 셋
linear = nn.Linear(784, 10, bias=True).to(device)

# loss function 정의 (Cross Entropy 함수 사용)
# Cross_entropy 함수는 softmax를 포함하는 라이브러리(함수)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28*28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print(f'Epoch:{epoch}, cost={avg_cost}')
print('Learning Finish')

with torch.no_grad(): # 테스트 시에는 graident 를 계산할 필요 없음
    X_test = mnist_test.test_data.view(-1, 28*28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print('Accuracy: ', accuracy.item())

    r = random.randint(0, len(mnist_test)-1)
    X_data = mnist_test.test_data[r: r+1].view(-1, 784).float().to(device)
    Y_data = mnist_test.test_labels[r: r + 1].to(device)

    single_prediction = linear(X_data)
    print(f'label={Y_data.item()}, Prediction={torch.argmax(single_prediction, 1).item()}')

    plt.imshow(mnist_test.test_data[r:r+1].view(28, 28), camp='Greys', interpolation='nearest')
    plt.show()