def binary_search(key, pl, pr, in_list):
    while 1:
        pc = (pl + pr) // 2

        if key == in_list[pc]:
            break

        elif key > in_list[pc]:
            pl = pc + 1

        elif key < in_list[pc]:
            pr = pc - 1
        else:
            pass

        if pl > pr:
            break  # return  -1

    return (pl, pr, pc)

def bin_insert_sort(in_list):
    for i in range(1, len(in_list)):

        key = in_list[i]
        pl = 0
        pr = i - 1
        pl, pr, pc = binary_search(key, pl, pr, in_list)

        if pl <= pr:
            pd = pc + 1

        else:
            pd = pr + 1

        for j in range(i, pd, -1):
            in_list[j] = in_list[j-1]
        in_list[pd] = key

    return 0

if __name__ == '__main__':
    tmp_list = [4, 6, 3, 7, 1, 4, 9, 8, 4]
    bin_insert_sort(tmp_list)
    print(tmp_list)
