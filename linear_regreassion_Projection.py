import numpy as np

# Ax = b
# projection 을 통해 근사값 (w, b)
A = np.array([[1, 1], [2, 1], [3, 1]])
b = np.array([2, 4, 6])

# projection 을 이용해서 근사해 구하는 수식
# x^ = (inv(A.T*A) * A.T) * b
# X^: 구하고자하는 최적해
# *: matrix multiplication
# .T: Transpose
# inv(inverse matrix): np. linalg.inv (numpy 에서 제공하는 inverse matrix 구하는 라이브러리)
# transpose(3x2) => 2x3 * 3x2 ==> 2x2 ==> inv(2x2) => 2x2 * 2x3 (2번째 A.T)
# ==> 2x3 * 3x1(y_train) ==> 2x1 (weight, bias)
# 역행렬 구하기 위한 조건
# - squared matrix (2x2)
# - det != 0 이어야한다.
# - 역행렬을 구하고자하는 matrix 의 rank 는 full rank 를 가져야 함

c = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)
print(c)
print(f'weight={c[0]}, bias={c[1]}')