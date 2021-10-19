def quick_sort(in_list):
    if len(in_list) <= 1: return in_list
    pivot = in_list[0]
    left, right = [], []
    for item in in_list[1:]:
        if item > pivot: right.append(item)
        else: left.append(item)

    return quick_sort(left) + [pivot] + quick_sort(right)

test_list = [5, 8, 4, 2, 6, 1, 3, 9, 7]
print(quick_sort(test_list))