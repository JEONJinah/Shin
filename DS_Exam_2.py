def quick_sort(in_list):
    if len(in_list) <= 1: return in_list
    pivot = in_list[0]
    left, right = [], []
    for item in in_list[1:]:
        if item > pivot: right.append(item)
        else: left.append(item)

    return quick_sort(left) + [pivot] + quick_sort(right)

test_list = [88, 23, 2, 4, 6, 8, 11, 8, 2]
print(quick_sort(test_list))