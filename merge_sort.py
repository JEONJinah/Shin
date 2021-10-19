def merge(left, right):
    v = list()
    i = 0
    j = 0
    """
    left: 1 2 8 9 
    right: 3 4 5 6
    1 vs 3 ==> 1 (1 append to v list) i++
    2 vs 3 ==> 1 2 (v list) i++
    8 vs 3 ==> 1 2 3 (v list) j++
    """
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            v.append(left[i])
            i += 1
        else:  # left[i] > right[i]
            v.append(right[j])
            j += 1
        if i == len(left): v = v + right[j:len(right)]
        if j == len(right): v = v + left[i:len(left)]

    print("After Sorting: ", v)
    return v

def merge_sort(v):
    if len(v) <= 1: return v   # 2개의 그룹으로 나누고 나누고... 한개가 남을 때 까지.
    m = len(v)//2  # 함수로 입력되는 배열을 두 그룹으로 나누기 위해.
    left = merge_sort(v[0:m])
    right = merge_sort(v[m:len(v)])

    print("Before Sorting: ", v)
    return merge(left,right)

if __name__ == "__main__":
    in_arr = [0, 5, 7, 6, 1, 2, 8, 3, 4, 9]
    out_arr = list()
    out_arr = merge_sort(in_arr)

    print(out_arr)
