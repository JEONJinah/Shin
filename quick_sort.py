# 주어진 리스트를 2 그룹으로 나누는 코드 작성

def quick_sort(in_list, left, right):  # 재귀호출로 작성시 함수의 매개변수도 변경되어야함.
    n = len(in_list)
    pl = left # 0
    pr = right #n - 1
    x = in_list[(left+right) //2 ]
    # x = in_list[n // 2]   # 재귀 호출로 해당 함수를 호출할 경우에는 (left, right)

    while pl <= pr:
        while in_list[pl] < x: pl += 1
        while in_list[pl] < x: pr -= 1
        if pl <= pr:
            in_list[pl], in_list[pr] = in_list[pr], in_list[pl]
            pl += 1
            pr -= 1

    if left < pr: quick_sort(in_list, left, pr)
    if pl < right: quick_sort(in_list, pl, right)

    # print(f'피벗보다 작은 그룹: {in_list[0: pl]}')
    # print(f'피벗보다 큰 그룹: {in_list[pr+1: n]}')

if __name__ == "__main__":
    x = [5, 8, 4, 2, 6, 1, 3, 9, 7]
    quick_sort(x, 0, len(x) - 1)

    print(x)