# 이중 for 문 곱셈
for i in range(2,10):
    for j in range(1, 10):
        print(i*j, end=" ")
    print(" ")

a = [1,2,3,4]
result = [num * 3 for num in a if num % 2 == 0]
print(result)

a = [(1, 3), (2, 4), (3, 5)]
for (f, r) in a:
    print(f + r)

marks = [90, 5, 67, 45, 80]
number = 0 #print(student num)
for points in marks:
    number += 1
    if points < 60:
        print(f'{number}인 학생은 불합격입니다.')
    else:
        print(f'{number}인 학생은 합격입니다.')

#Range
# (Start, end, step)
for i in range(0, 10):
    print(i, end=" ")
print(" ")
for i in range(0, 10, 2):
    print(i, end=" ")
print()

#list comprehension
list_a = [1, 2, 3, 4]
result = [num*3 for num in list_a]
print(result)

result = [num*3 for num in list_a if num% 2 == 0]
print(result)

result = [x * y for x in range(2,10)
    for y in range(1, 10)]
print(result)
