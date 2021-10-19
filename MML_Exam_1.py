
# 2x2 행렬 구하는 공식
# def matrixmult(A,B):
#     n = len(A)
#     C = [[0]*n for _ in range(n)]
#     for i in range(n):
#         for j in range(n):
#             for k in range(n):
#                 C[i][j] += A[i][k]*B[k][j]
#     return C
#
# A = [[2, 3], [4, 1]]
# B = [[5, 7], [6, 8]]
#
# print('A = ', A)
# print('B = ', B)
# C = matrixmult(A, B)
# print('C = ', C)

import numpy as np
import random

def gen_data(mat, n, m):
    for i in range(n):
        for j in range(m):
            mat[i, j] = random.randint(1, 9)
    return mat


def matmul(mat1, mat2, result):
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for t in range(len(mat2)):
                result[i][j] += mat1[i][t] * mat2[t][j]

    return result


if __name__ == '__main__':
    n = random.randint(3, 20)
    m = random.randint(3, 20)

    mat1 = np.zeros((n, m))
    mat2 = np.zeros((m, n))

    result = np.zeros((n, n))

    print(gen_data(mat1, n, m))
    print(gen_data(mat2, m, n))
    print(matmul(mat1, mat2, result))



