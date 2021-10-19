import torch

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