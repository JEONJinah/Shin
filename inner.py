# abs
print(-3, abs(-3), abs(3))

# ALL / ANY (AND / OR 유사한 동작)
list_a = [1, 2, 3, 0]
list_b = [1, 2, 3]

#  False, True, ...  예상..값 작성하면 좋을 것 같습니다
print(all(list_a), all(list_b), any(list_a), any(list_b), any([0, "", []]))


# Chr 내장함수 해볼 것

# dir: 객체 또는 자료형에서 가지고 있느 내장함수의 목록을 반환
list_var = [1, 2, 3]
str_var = "ABC"
dict_var = {'key': 'value'}
print(dir(list_var))
print(dir(str_var))
print(dir(dict_var))

# divmod: 나눗셈을 몫과 나머지를 반환하는 내장ㅎ ㅏㅁ수
# 함수로 작성
def div_mod(in_a, in_b):
    return ((in_a//in_b), (in_a % in_b))

print(divmod(7, 3))
print(div_mod(7, 3))

# (ENUM)
for i, name in enumerate(['body', 'foo', 'bar']):
    print(i, name)

# for(i = 0; i < max_num; i++)
# {
#     # operation
#     if array[i] > 10
# }

# filter | filter(func, iterable 자료형)
# Func: def, lambda
def func_positive(x):
    return x > 0

print(list(filter(func_positive, [7, 2, -3, -4, 1])))
print(list(filter(lambda x: x > 0, [7, 2, -3, -4, 1])))

# id 변수 파트에서 설명했으므로 Pass
# id: 객체의 메모리 주소를 반환
# input: input의 결과는 문자열이다...

# len: 객체의 length를 반환하는 내장함수로 대부분의 객체의 내장함수에
# define NUM_ARR_A  (100)
# for (i=0; i < NUM_ARR_A; i++)

# 모든 자료형에서의 형변환은 프로그램을 작성하면서 ... 실습해보는 것으로...

# map| map(func, iterable 자료형)
def two_times(x): return x * 2
in_list = [1, 2, 3, 4]
print(list(map(two_times, in_list)))
# 함수대신에 lambda 를 사용해도 됨.

# zip
list_1 = [1, 2, 3]
list_2 = [4, 5, 6]
list_3 = list(zip(list_1, list_2))
print(list_3, list_3[0][0], list_3[0][1])
print(list(zip("abc", "def")))