import torch
import time
from torch import cuda
from sklearn.datasets import load_boston
from PIL import Image
import numpy as np
from glob import glob

'''
# Boston 집값을 2D-Tensor 로 표현
# Feature: 13? 14rodml (Column으로 이루어져 있음)
boston = load_boston()  # load_boston 함수: numpy의 array type으로 반환
# Matrix (Row, Column): 506개의 row와 13개의 Column으로 이루어져 있음
print(boston.data.shape)

# numpy to pytorch Tensor
boston_tensor = torch.from_numpy(boston.data)
print(boston_tensor.size(), boston_tensor.dim())  # 예상결과: size(n x m: 506x13), dim:2

# 2개의 row( boston 2개의 집에 대한 정보 출력), column(1~11: 인덱스 기준)
# 13개의 feature: 0~12 (1~11)
# print(boston_tensor[row 조작, col 조작]) #","를 기준으로 row와 column 조작가능
print(boston_tensor[:2, 1:-1])

# 모든 특징을 포함한 2개의 row를 가지고 오고 싶을 때는,,,
print(boston_tensor[:2, :])
print(boston_tensor[:2])

# 3-dimensional Tensor (30-Tensor)
# gray scale 이미지 한 장 (1 channel image) * batch size
# 학습 진행 시 (영상의 입력 사이즈가 동일해야 함) example : 224x224x3
# gray scale 일 경우에는 224x224x1 (channel 1 대신에 batch size 를 넣는 것과 동일함)
# 학습 시에 한 번에 500장씩 메모리에 load 해서 matrix multiplication 하고자 할 때...
# 500 x 224 x 224로 load 해서 연산 수행
# 3 channel image 는 3D-Tensor, 이미지를 plot 하기 위해서, PIL 모듈로부터 Image 함수
# from PIL import Image
dog = np.array(Image.open('dog.4.jpg').resize((224, 224)))
dog_tensor = torch.from_numpy(dog)
# size 와 dim 을 한 번 확인
print(f'size and dimension of the dog color image are {dog_tensor.size()}, {dog_tensor.dim()}')

# 4-D Tensor
# Cat-Dog Download (추후에 classification 도 해보는 걸로..)
# 지금은 tensor 개념 및 연산에 대해 집중.
# /, ./ 다름@.@
data_path = "./archive/dataset/training_set/dogs/"
# data_path 에서 특정 문자열이 포함된 요소들을 리스트로 반환하고 싶을 경우
# 사용할 수 있는 라이브러리는?
# os.listdir(data_path) ==> 해당 폴더에 이미지 파일뿐만 아니라 폴더 내에 있는 모든 파일들을 리스트로 반환
# 특정 문자열을 포함한 데이터를 반환하는 library...
# dogs = lib(data_path + "*.jpg") # dogs 라는 변수에 리스트로 반환( 개 사진의 파일이름을 포함한 경로)
dogs = glob(data_path + "*.jpg")
# 예상되는 결과는, 해당 디렉토리 내에 jpg 파일의 파일 이름을 리스트로 반환
# 체크 (리스트에서 2개만 print)
print(dogs[:2])

# 64개의 데이터를 읽어서 np.array 에 저장 ==> torch.from_numpy(n-D Tensor)
img_list = []
# dogs 리스트에는 개의 모든 파일 경로 및 이름이 저장되어 있음
# 이 중에서 처음 64개만 가지고 오려고 함
for dog in dogs[:64]:
    img_list.append(np.array(Image.open(dog).resize((224, 224))))
dogs_imgs = np.array(img_list)

# list comprehension( 리스트 내포):
# 리스트 내에 for, if satatements를 사용해서 리스트로 반환 가능한 형식
# 위의 코드 3줄을 리스트 내포 방식을 이용하여 한 줄로 표현.(코드 작성)
# 작성을 잘 하시면 좋고, 아니면 꼭 이해라도 합시다.

[np.array(Image.open(dog).resize((224, 224))) for dog in dogs[:64]]

# 예상 결과 값: dim : 4, shape: (64x224x224x3), size = 다 곱한 값
print(dogs_imgs.ndim, dogs_imgs.size, dogs_imgs.shape)
# 상식으로 알고 갑시다.
# dogs_imgs = dogs_imgs.reshape(64, 224, 224, 3)
# 전체라는 의미로 자주 사용되는 숫자 -> -1
dogs_imgs = dogs_imgs.reshape(-1, 224, 224, 3)
dogs_tensor = torch.from_numpy(dogs_imgs)

# 백터, 행렬
# 행렬 A, B 가 있을 경우, A + B, A - B 를 하고자 할 때, 두 행렬의 크기가 같아야 함.
# 행렬의 크기가 다르더라도 자동으로 행렬의 크기를 조정하여 연산이 가능하도록 하는 방법
# [ Broadcasting ]
# Error 를 발생시키는게 제일 좋음. Wx+b 같은 연산을 수행할 때, W, b값 자체의 갯수가 상이할 수 있음
# 이럴 경우에 효율적으로 사용 가능한 방법
# pytorch 에서는 행렬의 크기가 다르더라도 Error 가 발생하지 않기 때문에 사용할 때 주의해야 함

# 1x2 행렬 두개를 생성하고 덧셈 수행
# 2D Tensor 생성
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([[2, 2]])
print(m1 + m2, (m1 + m2).shape)  # 예상 값 : [[5, 5]], 1x2 matrix

# Broadcasting
# m1 = 1x2, m2 = 1x1 벡터
m1 = torch.FloatTensor([[3, 3]])
m2 = torch.FloatTensor([2])  # 계산 시에는 [2] ==> [[2, 2]] 변환하여 계산 (내부적으로 Broadcasting 을 통해)
print(m1 + m2, (m1 + m2).shape)  # 동일하게 나오는 것을 확인

# m1 = 1x2 행렬, m2 = 2x1 행렬로 선언하고 덧셈 수행
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])

1x2, 2x1 case 에서는 두 행렬 모두 2x2로 변환 후 덧셈을 수행
m1 = [[1, 2],
      [1, 2]]
m2 = [[3],
      [4]]
m2 = [[3, 4],
      [4, 4]]   broadcasting 으로 변경됨

print(m1 + m2, (m1 + m2).shape)
'''
###########################

# in-place 연산(덮어쓰기 연산)
# 2x2 행렬에 해당하는 텐서 생성
a = torch.FloatTensor([[1, 2], [3, 4]])
b = torch.FloatTensor([[1, 2], [3, 4]])
# 덧셈을 하기 위해서 (a + b, torch.add(a, b))
# a+b (같은 값을 더한 것과 결과가 동일하기 때문에 결과는 a텐서 또는 b텐서의 두배 [[2, 4] , [6,  8]])
# 더하기 결과 값을 위해 새로운 객체가 내부적으로 생성되고 메모리에 할당됨(a,b 변수에는 변화가 ㅇ벗음)
print(f'{a+b}\n{torch.add(a, b)}\n2D-Tensor a: {a}')
# a+=1 과 같은 형태로 a가 가리키는 메모리 (계산된 결과 값에 해당하는 메모리)를 바로 반영
# Broadcasting + 인라인 덧셈(_)  언더바를 사용
# a.add_(2)
# 예상결과 값: [ [3, 4], [5, 6]]
print(f'{a.add_(2)}\n2D-Tensor a: {a}')

# 자주 사용되는 기능들에 대해서 실습
# 1) Matrix Multiplication vs. Multiplication (Element-wise 곱: 원소별 곱셈)
# 행렬 곱(Matrix Multiplication): t1.matmul(t2), torch.matmul(t1, t2)
# element-wise 곱: m1 * m2, or m1.mul(m2)
# 행렬 곱 실습
m1 = torch.FloatTensor([[1, 2], [3, 4]]) # 2x2 행렬
# 행렬 곱을 정상적으로 수행하기 위해서는 m2의 행렬은
# m2 = 2 x n (2x5)
m2 = torch.FloatTensor([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]])
print(m1.shape, m2.shape, m1.matmul(m2), torch.matmul(m1, m2)) # 2x2 matmul 2x5 = 2x5

# element-wise 곱
# element-wise 곱: m1 * m2, or m1.mul(m2)
m2 = torch.FloatTensor([[1], [2]])
# Broad casting 사용시: 2x5 와 같이 차수가 큰 것을 broad cast를 하면 Error: 어떤 item (요소값),
# 을 줄여야 되는지 정의되지 않았기 때문
# m1 * m2

# m1 = 2x2
# m2 <-- 2x1를 broad cast로 element wise곱을 했을 때와
# m2 <-- 1x2를 broad cast로 element wise곱을 했을때 결과는...
"""
2x1 broad cast
[ [1], 
  [2] ] 
[ [1, 1],
  [2, 2] ]
"""
print(m1*m2, m1.mul(m2))

# CPU vs. GPU 처리 속도 측정
# 딥러닝의 모델을 학습하기 위해서 가장 많이 수행하는 연산 중에 한 가지가 matrix multiplication
# import time
# from torch import cuda
start = time.time()
a = torch.rand(20000, 20000)
b = torch.rand(20000, 20000)
c = a.matmul(b)
end = time.time()
print(f'CPU: {end-start}')

# GPU 사용할 경우(import cuda)
use_gpu = cuda.is_available()
print(torch.cuda.device_count())
if use_gpu:
    print("using CUDA")
    a = a.cuda() # a matrix 사용 시 gpu 자원 활용
    b = b.cuda()
    a.matmul(b)
end = time.time()
if use_gpu: print(f'GPU: {end-start}')

# 2) 평균 ( Mean) 실습
# 요소 값이 두 개인 1-D Tensor의 평균 (vector)
t2 = torch.FloatTensor([1, 2]) # t2.mean = (1 + 2) /2 (원소개수)
# 3-dimensional vector, 1-D Tensor의 평균
t3 = torch.FloatTensor([1, 2, 3])
print(t2.mean(), t3.mean())
# matrix: 2D Tensor의 평균을 구할 때(2x2 행렬)
t = torch.FloatTensor([[1, 2], [3, 4]])
"""
[ [1 2], 
  [3 4] ]
평균은 행렬에 있는 모든 요소를 더한 값을 갯수로 나눈 값
1+ 2+ 3+ 4/ 4 = 2.5
"""
print(t.mean())

# mean 함수에서 parameter  중에 dimension 옵션
# t.mean(dim=0), t.mean(dim=1)
# dim=0: 0이라는 것은 첫번째 차원 (2D-Tensor, 행렬이고, row, column 순서 ==> 0: row, 1:column을 의미)
# ==> 0번째 차원(row)을 제거(배제?, 무시?)하고 평균을 계산함
# dim=1: 컬럼을 배제하고 평균 계산
# 파라미터에 dim=0을 사용하면: column 별 평균을 구함
# 파라미터에 dim=1을 사용하면: row 별 평균을 구함
# t = [1, 2] [3, 4]
print(t.mean(dim=0)) # (1, 3) (2, 4)의  각각의 평균값
print(t.mean(dim=1)) # (1, 2), (3, 4)의 각각의 평균값

# 3) 덧셈 (sum 함수 사용하면 됨, 여기에서도 dif 파라미터를 사용할 수 있음)
print(t.sum())
print(t.sum(dim=0))  # 예상되는 결과 값:
print(t.sum(dim=1))  # 예상되는 결과 값:

# 4) max. argmax 실습
# max: Tensor의 원소 중에 가장 큰 값을 반환
# argmax: Tensor 의 원소 중에 최댓값에 해당한느 인덱스를 반환하는 함수
# 밑에 정의한 matrix 에서 index : 0번 인덱스는 1, 1번 인덱스는 4, 2번 인덱스는 5, 3번 인덱스는 3
t = torch.FloatTensor([[1, 4], [5, 3]])
print(f'argmax: {t.argmax()}')
# t.max 에서도 dimension 파라미터가 존재
# t.max 의 dim param 을 사용하면 argmax 와 동일한 역할 가능함
# t.max(dim=0): (1, 5), (4, 3) ==> max:5, 4 argmax: 1, 0
# t.max(dim=1): (1, 4), (5, 3) ==> max: 4, 5 argmax: 1, 0
print(f'{t.max(dim=0)}\n{t.max(dim=1)}')

"""
view is same as reshape (numpy, tensorflow, etc.)
"""
# 5) view(뷰): reshape 과 동일하다고 생각하면 됨
# 원소의 수를 유지하면서 텐서의 크기만 변경 (shape 만 변경)
# numpy, etc. 의 reshape 과 동일(텐서의 크기, 차수 등은 변경되지만, 원소의 총 갯수는 변화가 없음)
# 2x2x3 tensor 를 생성(정의)
t = torch.FloatTensor([
    [
        [0, 1, 2],
        [3, 4, 5]
    ],
    [
        [0, 1, 2],
        [3, 4, 5]
    ]
])
print(t.shape, t.dim())

# 5-1) 3D-Tensor ==> 2D Tensor 로 변경
# t2 = t.reshape ==> view
# -1은 전체를 의미,,
# range, index등에 사용될 시에는 인덱스를 뒤에서 부터 -1씩 감소한다는 의미
# + 첫번째 차원으 파이토치(numpy, tensorflow, etc.)에서 알아서 계산해줘라는 의미를 가짐
# 2D Tensor == matrix 에서의 첫번째 차원은 row가 되고, row는 파이토치에서 알아서 계산하시오.
t2 = t.view([-1, 3])
print(t2, t2.shape)

# 5-2) 3D Tensor ==> 3D Tensor 변경(차수 변경을 하고 싶음: 2x2x3 => nx1x3)
# n = 값이 무엇인가요? 총 원소의 갯수는 달라지지 않으므로 2x2x3 = 12 이고, nx1x3=12가 나와야하므로
# n = 4 ==> 4x1x3
t3 = t.view([-1, 1, 3])
"""
t3 = [
 [
    [0, 1, 2]
 ],
 [
    [1, 2, 3]
 ],
 [
    [0, 1, 2]
 ],
 [
    [1, 2, 3]
 ]
]
"""
print(t3, t3.shape)

# 6) squeeze : 차원이 1인 것을 제거
# (n x 1 행렬 ==> ㅜ dimensional 한 vector로 변환)
# n x 1인 행렬을 요소가 n개인 벡터로 변환한다는 것과 같은 의미
t = torch.FloatTensor([[0], [1], [2]]) # 3x1 matrix
t_sq = t.squeeze()
print(f't={t}\nt_sq={t_sq}\nt.shape={t.shape}\nt_sq.shape={t_sq.shape}')

# 7) unsqueeze : 특정 위치(row, column, etc.) 에 1인 차원을 추가
t = torch.FloatTensor([0, 1, 2]) # 요소가 3개인 벡터 (3-dimensional vector) size ([3])
# 1x3 , 3x1
# dim=0, 1(0: row 에 추가, 1: column 추가)
# 1x3 (즉, row 에 1을 추가)
t_row = t.unsqueeze(0) # t_row = [[0, 1, 2]]
# 3x1 (column 1 추가)
t_col = t.unsqueeze(1) # t_col = [[0], [1], [2]]
print(f'unsqueeze test: ', t.shape, t_row.shape, t_col.shape)
print(t_row, t_col)

# 8) Concatenate: Tensor 끼리 연결을 하고자 할때
# x, y: 2x2 tensor 정의(생성) ==> x, y tensor 를 row 방향 또는 column 방향으로 연결
# torch.cat 사용해서 연결 (cat = concatenate)
# option dim=0 일 경우, row를 기준으로 텐서들을 연결==> 2x2에서 4x2 행렬로 concat (row 차수 증가)
# dim=1일 경우, column을 기준으로 텐서들을 연결 ==> 2x2에서 2x4 행렬로 concat (column 차수 증가)
x = torch.FloatTensor([[1, 2], [3, 4]])
y = torch.FloatTensor([[5, 6], [7, 8]])
t_row_concat = torch.cat([x, y], dim=0)  # 예상 값:  [ [1, 2], [3, 4], [5, 6], [7, 8] ]
t_col_concat = torch.cat([x, y], dim=1)  # 예상 값:  [ [1, 2, 5, 6], [3, 4, 7, 8] ]

print(t_row_concat, "\n", t_row_concat.shape, "\n", t_col_concat, "\n", t_col_concat.shape)

# 9) Stacking: 텐서들을 연결하는 다른 방법
# 스택킹에는 많은 함수들을 내포하고 있는 모듈
# 2-dimensional 한 벡터를 3개 생성, stacking 을 통해서 3x2 tensor 생성
x = torch.FloatTensor([1, 2])  # 1D Tensor [2,] [1x2] or [2x1]
y = torch.FloatTensor([3, 4])
z = torch.FloatTensor([5, 6])
t_con = torch.cat([x, y, z])
print(t_con, t_con.shape)
# 예상 값:  [ [1, 2], [3, 4], [5, 6], [7, 8] ]
print(x.unsqueeze(0))  #  [2] --> [1x2]
print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)]))
# stack
print(torch.stack([x, y, z], dim=0))
print(torch.stack([x, y, z], dim=1))

# 10) Tensor 들을 1 또는 0으로 초기화.. [ones_like, zero_like]
x = torch.FloatTensor([
    [0, 1, 2],
    [3, 4, 5]
])

x_ones = torch.ones_like(x)  # 예상 값: 2x3 형태의 2D 텐서의 요소 값이 모두 1인 tensor
x_zeros = torch.zeros_like(x)

print(f'{x}\n{x_ones}\n{x_zeros}')