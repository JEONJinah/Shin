# 상속과 관련된 실습
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Computer:
    def __init__(self, cpu, ram):
        self.cpu = cpu
        self.ram = ram

    def browse(self):
        print("Web 서핑..")

    def calc(self):
        print("Calcuate CV, ML, DL...")


# 특정 class 를 상속 받기 위해서는 클래스명 뒤에 (상속받을 클래스명)


class Laptop(Computer):

    def __init__(self, cpu, ram, battery, ):
        # 자식 클래스에서 부모 클래스의 내용을 그대로 사용하고 싶을 때
        # 컴퓨터 클래스의 __init__ 메소드를 그대로 사용하고 싶음
        super().__init__(cpu, ram)  # 이걸 쓰면 밑에 두 줄은 노쓸모!!
        # self.cpu = cpu
        # self.ram = ram
        self.battery = battery

    def move(self):
        print("노트북은 이동이 용이함")

# Example#2 : Rectangle 넓이, 둘레 등 구하기 ( Rect > squared rect)


class Rect:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h

    def perimeter(self):     # 둘레!!
        return 2 * self.w + 2 * self.h

    # area_tmp 추가
    def area_tmp(self):
        print("inh-->inh: 상속에 상속")
        return self.w * self.h


class Square:
    def __init__(self, w):
        self.w = w

    def area(self):
        return self.w ** 2

    def perimeter(self):     # 둘레!!
        return 4 * self.w


class SquareIhn(Rect):
    def __init__(self, w):
        super().__init__(w, w)

    def area_tmp(self):
        print("inh : 바로 위에 클래스 상속")
        return self.w * self.w


# super() vs. super(SquareInh, self)
# 탐색 범위가 달라짐 (다중 상속 또는 상속 - > 상속일 경우)
# Rect, SquareIhn(area_tmp 메소드를 이용해서 결과 확인)

class Cube(SquareIhn):
    # 정육면체의 전체 면적 : w * h * 6 (w*w*6)
    def surface_area(self):
        sur_area = super(SquareIhn, self).area_tmp()
        return sur_area * 6

    def volumn(self):
        vol = super().area_tmp()
        return vol * self.w


# 단순 선형 회귀 클래스 구현
class LinearRegressionModel(nn.Module):     # torch.nn.Module을 상속 받음
    def __init__(self):
        # nn.Module 클래스의 속성들을 가지고 초기화
        super().__init__()
        # model = nn.Linear(1, 1)
        self.Linear = nn.Linear(1, 1)

    def forward(self, train_x):
        return self.Linear(train_x)     # WX 값 리턴


if __name__ == '__main__':
    laptop = Laptop("Intel CPU 2G", "8GB", "100%")
    laptop.browse(), laptop.move()

    squar_inh = SquareIhn(4)

    # 예상하는 넓이와 둘레의 결과 값 : 넓이 =16
    print(squar_inh.area(), squar_inh.perimeter())

    cube = Cube(3)
    print(cube.surface_area())
    print(cube.volumn())


    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[1], [2], [3]])

    # previous version :  model = nn.Linear(1, 1)
    # 객체를 생성한다는 것 이외에는 동일함.
    model = LinearRegressionModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    nb_epochs = 2000
    for epoch in range(nb_epochs):
        pred = model(x_train)
        cost = F.mse_loss(pred, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if epoch % 100 == 0:
            tmp = list(model.parameters())
            print(f'Epoch: {epoch:4d} Cost: {cost.item(): .6f}')

print(f'w, b: {tmp[0]}, {tmp[1]}')


# a = Computer("1.6", "8G")

# 실제 학습 시키는 부분은 딥러닝 수학시간에 했던 것을 참고해서 동작여부 확인해보세요.
