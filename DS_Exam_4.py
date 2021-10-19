def quick_sort(in_list, left, right):
    n = len(in_list)
    pl = left
    pr = right
    x = in_list[(left+right) // 2]


    while pl <= pr:
        while in_list[pl] < x: pl += 1
        while in_list[pl] < x: pr -= 1
        if pl <= pr:
            in_list[pl], in_list[pr] = in_list[pr], in_list[pl]
            pl += 1
            pr -= 1

    if left < pr: quick_sort(in_list, left, pr)
    if pl < right: quick_sort(in_list, pl, right)


if __name__ == "__main__":
    x = [5, 8, 4, 2, 6, 1, 3, 9, 7]
    quick_sort(x, 0, len(x) - 1)

    print(x)