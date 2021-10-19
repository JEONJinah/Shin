# 7 5 9 0 3 1 6 2 4 8
# step 1
# 7은 정렬되어 있다고 가정, 두번째 데이터 5가 어떤 위치로 들어갈지 판단 (7의 왼쪽 혹은 오른쪽)
# 7은 왼쪽에 삽입
'''
# 리스트로 입력받기
# unsorted_list = [int(x) for x in input().split()] (밑에 세 줄을 간추린 것)
unsorted_list = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
for x in input().split():
    unsorted_list.append(int(x))

# insertion sorting 알고리즘
def insertion_sort(unsorted_list):
    j = 1
    for j in range(j, len(unsorted_list)):
        key = unsorted_list[j]
        i = j - 1   # 바로 이전 값에 위치를 가리키고
        while i >= 0 and unsorted_list[i] > key:  # 몇번째 자리에 값을 넣어야하는지 찾는 과정
            unsorted_list[i+1] = unsorted_list[i]  # 조건을 만족하면 swap
            i = i - 1
        unsorted_list[i+1] = key

    return unsorted_list

print(unsorted_list)

def ins_sort(a):
    n = len(a)
    for i in range(1, n):
        key = a[i]
        j = i - 1

        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key

d = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
ins_sort(d)
print(d)

# 삽입정렬
def insertion_sort(in_list):

    for i in range(1, len(in_list)):
    # i가 실제는 n번째 인덱스에 위치해 있을 때
    # n~0번째까지 탐색하면서 작은 값이 있으면 swap
        print(in_list)
        for j in range(i, 0, -1):
            if in_list[j] < in_list[j - 1]:
                temp = in_list[j]
                in_list[j] = in_list[j - 1]
                in_list[j - 1] = temp
            else:  # 자기보다 작은 데이터를 만나면 그 위치에서 멈추기 위해
                break
if __name__ == "__main__":
    list_a = [7, 5, 9, 0, 3, 1, 6, 2, 4, 8]
    insertion_sort(list_a)
    print(list_a)

# 이진삽입정렬 1
def binary_insertion_sort(a):
    n = len(a)
    for i in range(1, n):
        key = a[i]
        pl = 0
        pr = i - 1

        while True:
            pc = (pl + pr) // 2
            if a[pc] == key:
                break
            elif a[pc] < key:
                pl = pc + 1
            else:
                pr = pc - 1
            if pl > pr:
                break

        if pl <= pr:
            pd = pc + 1
        else:
            pd = pr + 1
        print(f'pl: {pl}, pr: {pr}, pd: {pd}')
        for j in range(i, pd, -1):
            a[j] = a[j-1]

        a[pd] = key


print('이진 삽입 정렬')
num = int(input('원소 수 입력: '))
x = [None] * num    # 원소 수가 num인 배열을 생성

for i in range(num):
    x[i] = int(input(f'x[{i}]: '))

print(''.join(str(x)))
binary_insertion_sort(x)
print(''.join(str(x)))
'''
# 이진 삽입정렬 2
import bisect
def binary_insertion_sort(a):
    for i in range(1, len(a)):
        bisect.insort(a, a.pop(i), 0, i)
        # insort(a, x, lo, hi)

print('이진 삽입 정렬')
num = int(input('원소 수 입력: '))
x = [None] * num    # 원소 수가 num인 배열을 생성

for i in range(num):
    x[i] = int(input(f'x[{i}]: '))

print(''.join(str(x)))
binary_insertion_sort(x)
print(''.join(str(x)))