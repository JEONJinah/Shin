# # 딕셔너리 연습
# a = {1: 'a', 2: 'b'}
# a[3] = 'c'
# a['last_name'] = ['Shin', 'Kim', 'Lee']
# del a[1]
# print(a)
# a = {"학과": ["AI_Eng", "Smart Elect"], "num_of_student": [22,60]}
# print(a)
#
# a = {
#     ('name', 'age'): (["shin","kim"],[25, 30])
# }
# print(a)

# # 딕셔너리 functions
# a = {
#     'name' : ['김미경', '차혜인', '이연우', '전진아'],
#     'ID' : [21000, 21001, 21002, 21003]
# }
# list_tmp = ['name', 'ID']
# dict_keys = a.keys()
# list_tmp.append('phone_num')
# dict_list = list(a.keys())
# dict_list.append('phone_num')
# print(dict_list)
# print(dict_keys)
#
# b = "123"
# c = int(b)
# print(type(b), type(c), b, c)
#
# b = 12
# c = 4
# print(b/c)

# # 딕셔너리 예제 3 자료형
# # 집합 자료형 연습
# s1 = set([3, 1, 4, 2])# 순서가 없고
# s2 = set("Hello") # 순서가 없고 중복을 허용하지 않는다.
# print(s1, s2)
#
# # Set Function (교집합, 합집합, 차집합)
# s1 = set([1, 2, 3, 4, 5])
# s2 = ([2, 3, 5])
# print(f'교집합={s1.intersection(s2)}, 합집합={s1.union(s2)}, 차집합={s1.difference(s2)}')
#
# s1 = set([2, 3, 5])
# s1.update([5, 7, 0])
# print(s1)

# bool 자료형 연습
a = True
b = False
print(type(a), type(b), f'1 is same as 1?{1==1}, 1==2입니까?{1==2}, 1은 2보다 작나요?{1<2}')
str_t = "Hello" #Hello > Hell > Hel > He > H
idx = 1
while str_t:
    print(idx, len(str_t), str_t)
    str_t = str_t[:len(str_t) - idx]

if []:
    print("True")
else:
    print("False")

if "python":
    print("True")
else:
    print("False")

if "":
    print("True")
else:
    print("False")

if ():
    print("True")
else:
    print("False")

print(bool([1]))

if 0:
    print("True")
else:
    print("False")

if None:
        print("True")
else:
    print("False")
