# def selection_sort(arr):
#     for i in range(len(arr)-1):
#         min_index = i
#         # 최소값 찾는 처리
#         for k in range(i+1, len(arr)):
#             if arr[k] < arr[min_index]:
#                 min_index = k
#         # 최소값의 위치를 바꿔주는 처리
#         tmp = arr[i]
#         arr[i] = arr[min_index]
#         arr[min_index] = tmp
#     return arr
#
# test = [3, 81, 4, 76, 5, 25, 34]
# print(selection_sort(test))

# Select Sorting
'''
Description
주어진 리스트(배열)에서 가장 작은 값을 찾은 후, 정렬되지 않은 부분의 제일 앞으로 옮기는 방법
==> 예를들어 1st iteration  이면
첫번째 인덱스의 값[value:77, index:0]과 선형 검색을 통해서 찾은 최소 값[value: 0, index: 17]을 swap
[value: 0, index: 0] ..... [value: 77, index: 17]
제일 작은 값을 찾아서 한 번 옮긴다는 것은 현재 위치의 앞쪽 까지는 이미 정렬이 되어 있다는 의미
마지막 남은 요소까지 검사를 하고 나면 정렬이 다 된 것이므로 프로그램 종료
'''


def select_ascending_sorting(in_list):
    n = len(in_list)

    # 리스트 갯수만큼 반복문에 있는 구문(operation)을 수행
    # n이 아니라 n-1인 이유 ==> 리스트에 인덱싱이라는 방법을 통해 값을 가지고 오기 위해서!!
    # (인덱스는 1이 아닌 0부터 시작) // i는 인덱스를 의미
    """
    for i in range(10)
    --> range(10) : iteration마다 i에 대입되는 값은(인덱스): 0 1 2 3 4 5 6 7 8 9
    --> range(10-1) : 0 1 2 3 4 5 6 7 8 

    데이터가 10개라고 가정
    i = 0 1 2 3 4 5 6 7 8 
    j =   1 2 3 4 5 6 7 8 9 

    i = 1 2 3 4 5 6 7 8
    j =   2 3 4 5 6 7 8 9
    """
    # 오름차순으로 단순선택정렬
    for i in range(n - 1):
        min = i  # selext sorting은 리스트에서 최솟값의 인덱스를 알아야, 현재 인덱스 값과 swap 가능
        for j in range(i + 1, n):
            if in_list[j] < in_list[min]:
                min = j
        in_list[i], in_list[min] = in_list[min], in_list[i]

    return in_list


def select_descending_sorting(in_list):
    n = len(in_list)

    # 내림차순으로 단순선택정렬
    for i in range(0, n - 1):
        max = i
        for j in range(i + 1, n):
            if in_list[j] > in_list[max]:
                max = j
        in_list[i], in_list[max] = in_list[max], in_list[i]

    return in_list


if __name__ == '__main__':
    # 임의에 정렬되지 않은 리스트 생성
    # 작성한 선택정렬 알고리즘의 입력으로 해당 리스트 넣으면
    # 정렬되어진 리스트가 출력되어지기를 원함
    list_tmp = [99, 17, 22, 13, 0, 9, 1, 3, 44, 2, 1, 22, 4]
print(select_ascending_sorting(list_tmp))
print(select_descending_sorting(list_tmp))
'''
    s = [10, 45, 34, 65, 3, 60, 90, -4, 40]
    c = len(s)

    for i in range(0, c - 1):
        for j in range(i + 1, c):
            if s[i] < s[j]:
                s[i], s[j] = s[j], s[i]
    print(s)
'''
from typing import MutableSequence

def insertion_sort(a: MutableSequence) -> None:

    n = len(a)
    for i in range(1, n):

        j = i
        tmp = a[i]
        while j > 0 and a[j - 1] > tmp:
            a[j] = a[j - 1]
            j -= 1
        a[j] = tmp

if __name__ == "__main__":
    print('단순 삽입 정렬을 수행합니다.')
    num = int(input('원소 수를 입력하세요.: '))
    x = [None] * num

    for i in range(num):
        x[i] = int(input(f'x[{i}]: '))

    insertion_sort(x)

    print('오름차순으로 정렬했습니다.')

def insert_sort(in_arg):
    # for/while(반복문)의 경우에는 index, index 의 value 를 가지고 오기 위해서
    # 리스트의 전체 길이를 알고 있으면 사용하기 용이합
    n = len(in_arg)


    # 입력 리스트의 갯수가 10이면
    # range(1, n): 1, 2, 3, 4, 5, 6, 7, 8, 9
    for i in range(1, n):
        tmp = in_arg[i] # in_arg 리스트의 i번째 인덱스에 있는 value 를 tmp 로 assign
        j = i

        '''
        while statement: 내에 존재하는 condition 이 참일 경우에 계속 loop 수행하는 것
        while 문 뒤에 있는 내용은 조건문을 의미
        and: 아래 작성된 코드에서는 조건이 2개이므로 2개의 조건이 모두 "참"일 경우에만 loop 수행
        조건이 하나라도 false 이면 while 문을 빠져나오게됨
        while 문이 참일 경우 수행하는 것은 
        - 적절한 위치에 tmp 값을 넣기 위해 하나씩 옆으로 이동하는 것
        while 문을 빠져나오게 되면
        - 해당 인덱스에 tmp 값을 삽입
        '''
        while j > 0 and in_arg[j - 1] > tmp:
            in_arg[j] = in_arg[j - 1]
            j -= 1
        in_arg[j] = tmp

    print(in_arg)
    # 다른 프로그램에서 현재 작성한 sorting 알고리즘을 import 해서 사용할 경우
    # 아래와 같이 return 문 사용하면 됨
    # return in_arg

if __name__ == '__main__':
    list_tmp = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
    insert_sort(list_tmp)
