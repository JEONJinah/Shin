import torch
import numpy as np
from sklearn.datasets import load_boston
from PIL import Image
from glob import glob

print(torch.__version__)

# vector : 요소들을 나열한 것 (pytorch : 1-dimensional Tensor or 10 Tensor)
# 텐서를 만들기 위해서는 torch.FloatTensor()
# 괄호 안에 1차원, 2차원 배열을 넣음으로써 차원을 결정 (입력되는 배열은 numpy와 유사함)
# 지난 주의 평균 온도를 벡터에 저장
temp = torch.FloatTensor([15, 17, 19.2, 22.3, 20, 19, 16]) # 1차원 배열 형태로 작성
print(temp.size(), temp.dim())
# Tensor에서도 python의 indexing, slicing 사용 가능
# indexing, slicing
print(f'월, 화 평균 온도는 : {temp[0]}, {temp[1]}입니다.')
print(f'화~목요일의 평균 온도는 : {temp[1:4]}입니다.')

# matrix : 20 (2-dimensional) Tensor
# pytorch에서 2d tensor 생성 (4x3 matrix, 4x3만 2차원 tensor)
# 2차원 작성 시 ([[], [], [], []])
t = torch.FloatTensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [10, 11, 12]
])
print(t.size(), t.dim())
# slicing
# 첫번째 차원의 모든 값과 두번째 차원의 두번째 요소만 가지고 오고 싶을 때
# 모든 행에서 두번째 열의 요소만 가지고 오고 싶음 (2, 5, 8, 11)
print(t[:, 1])
# 첫번째 차원의 3번째 요소, 두번째 차원의 모든 것
# 3행의 모든 요소
print(t[2, :])
# 4x3 matrix (20 tensor)
"""
[5, 6]
[8, 9]
"""
# 2,3번째 행에서 1,2번째 열을 가져오고 싶을 때
print(t[1:3, 1:3])
print(t[1:3, 1:])

# Boston 집값을 2D-Tensor 로 표현
# Feature: 13? 14rodml (Column으로 이루어져 있음)
boston = load_boston() # load_boston 함수: numpy의 array type으로 반환
 # Matrix (Row, Column): 506개의 row와 13개의 Column으로 이루어져 있음
print(boston.data.shape)

# numpy to pytorch Tensor
boston_tensor = torch.from_numpy(boston.data)
print(boston_tensor.size(), boston_tensor.dim()) # 예상되는 결과: size(n x m: 506x13), dim: 2)

# 2개의 row(보스턴 2개의 집에 대한 정보 출력), column(1~11: 인덱스 기준)
# 13개의 feature: 0~12 (1~11)
# print(boston_tensor[row 조작, col 조작]) # "," 기준으로row와column조작 가능
'''
슬라이싱 [시작인덱스:종료인덱스]
시작인덱스와 종료인덱스는 표기 안해도 됨
a[:2] ==> 시작인덱스를 표기하지 않았으므로 시작 인덱스는 0이다.a[0:2]
종료인덱스도 시작인덱스와 동일[표기를 안하면 전체(마지막인덱스)]

boston_tensor[:2] row의 시작은 0(표기하지 않았으므로), 종료인덱스는 2: 2개의 row를 가지고 와라...
boston_tensor[:], [여기는 column에서 짤라오고 싶음]
'''
print(boston_tensor[:2, 1: -1])
# 모든 특징을 포함한 2개의 row를 가지고 오고 싶을 때는...
print(boston_tensor[:2, :])
print(boston_tensor[:2])

# 3-dimensional Tensor (3D-Tensor)
# gray scale 이미지 한장 (1 channel image) * batch size
# 학습 진행 시 (영상의 입력 사이즈가 동일해야 함) example: 224x224x3 (3D Tensor)
# gray scale일 경우에는 224x224x1 (channel 1 대신에 batch size를 넣는 것과 동일함)
# 학습시에 한 번에 500장씩 메모리에 load해서 matrix multibplication 하고자할 때..
# 500 x 224 x 224로 load해서 연산 수행
# 3 channel image는 3D-Tensor, 이미지를 Plot 하기 위해서. PIL 모듈로부터 Image 함수 import
# from PIL import Image
# n-dimensional aray 반환(np.array 를 이용하여)
dog = np.array(Image.open("dog.181.jpg").resize((224,224)))
dog_tensor = torch.from_numpy(dog)
# size와 dim을 한 번 확인
print(f'size and dimension of the dog color image are {dog_tensor.size()}, {dog_tensor.dim()}')

# 4-D Tensor
# Cat-Dog Download (추후에 classification 도 해보는걸로...)
# 지금은 tensor  개념 및 연산에 대해 집중.
# /, ./ 다르다.
data_path = './dataset/training_set/dogs/'
# data_path에서 특정 문자열이 포함된 요소들을 리스트로 반환하고 싶을 경우
# 사용할 수 있는 라이브러리는?
# os.listdir(data_path) --> 해당 폴더의 이미지 파일 뿐만 아니라 폴더 내의 모든 파일들을 리스트로 반환
# 특정 문자열을 포함한 데이터를 반환하는 library
# dogs = lib(data_path + "*.jpg") # dogs라는 변수에 리스트로 반환 (개 사진의 파일 이름을 포함한 경로)
dogs = glob(data_path + "*.jpg")
# 예상 결과 : 해당 디렉토리 내 jpg 파일의 파일 이름을 리스트로 변환
# 체크 (리스트에서 2개만 print)
print(dogs[:2])


# 64개의 데이터를 읽어서 np.array에 저장 ==> torch,from_numpy (n-D Tensor)
img_list = []
# dogs 리스트에는 개의 모든 파일 경로 및 이름이 저장되어 있음
# 이 중에서 처음 64개만 가지고 오려고 함
for dog in dogs[:64]:
    img_list.append(np.array(Image.open(dog).resize((224, 224))))
dog_imgs = np.array(img_list)

# 파이썬은 list 내포(comprehension) : 리스트 내에 for, if statements를 사용해서 리스트로 반환 가능
# 위의 for문 3줄을 한 줄로 표현
dogs_imgs = np.array([np.array(Image.open(dog).resize((224, 224))) for dog in dogs[:64]])

# 결과값 : dim-4, size-다 곱한 값, shape-(64, 224, 224, 3),
print(dog_imgs.ndim, dog_imgs.size, dog_imgs.shape)
# 상식으로 알고 갑시다.
# dogs_imgs = dogs_imgs.reshape(64, 224, 224, 3)
# 전체..라는 의미로 자주 사용되는 숫자
dogs_imgs = dogs_imgs.reshape(-1, 224, 224, 3)
dogs_tensor = torch.from_numpy(dog_imgs)

# 벡터, 행렬
# 행렬 A, B가 있을 경우 A+B, A-B를 하고자 할 때, 두 행렬의 크기가 같아야 함.
# 행렬의 크기가 다르더라도 자동으로 행렬의 크기를 조정하여 연산이 가능하도록 하는 방법
# [ Broadcasting ]
# Error를 발생시키는게 제일 좋음. Wx+b 같은 연산을 수행할 때, w, b값 자체가 갯수가 상이할 수 있음
# 이럴 경우에 효율적으로 사용가능한 방법
# pytorch 에서는 행렬의 크기가 다르더라도 Error가 발생하지 않기 때문에 사용할 때 주의해야함

# 1x2 행렬 두개를 생성하고 덧셈 수행
# 2D Tensor 생성
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2, (m1+m2).shape) # 예상 값: [[5, 5]], 1x2 matrix

# BroadCasting
# m1 = 1x2, m2 = 1x1  벡터 덧셈
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([2]) # 계산시에는 [2] ==> [[2, 2]] 변환하여 계산(내부적으로 Broadcasting 을 통해)
print(m1 + m2, (m1+m2).shape) # 예상 값: [[5, 5]], 1x2 matrix (동일하게 나오는 것을 확인)

# m1 = 1x2 행렬, m2 = 2x1 행렬로 선언하고 덧셈 수행
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
'''
1x2, 2x1 case에서는 두 행렬 모두 2x2로 변환 후 덧셈을 수행
m1 = [ [1, 2], 
       [1, 2] ]
m2 = [ [3],
       [4] ]
m2 = [ [3, 3]
       [4, 4] ] 브로드캐스팅으로 변경됨    
'''
print(m1 + m2, (m1+m2).shape)

