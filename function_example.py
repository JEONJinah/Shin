# 함수 정의
# 더하기 함수 만들기
def add(a, b):
    # result = a + b
    # return result
    return a + b
print(f'3+4 = {add(3, 4)}')

# 함수는 총 4가지 type
# 입출력, 출력, 입력, XX

# 출력만 존재하는 함수
def add_5_6():
    return 5+6

rtn_value = add_5_6()
print(rtn_value)

#입력, 출력이 존재하지 않는 함수
def mul(a, b):
    print(a * b)

rtn_mul = mul(3, 4)
print(f'return value is {rtn_mul}')

# 입력도 출력도 없는 함수 작성
import math
def print_PI():
    print(math.pi)
rtn_pi_value = print_PI()

mul(3, 4), mul(a=3, b=4), mul(b= 44, a=22)

# 가변 인자 함수 작성 (입력된 데이터의 누적합산 결과 출력)
def add_many(*args):
    result = 0
    for i in args:
        result += i # result = result + i
    return result

rtn_val = []
rtn_val.append(add_many(1, 2, 3, 4, 5))
rtn_val.append(add_many(1, 2, 3, 4))
rtn_val.append(add_many(1, 2, 3))
print(rtn_val)

val_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
tmp = add_many(*val_list)
print(tmp)

def add_mul(operation, *args):
    if operation == 'add':
        result = 0
        for i in args:
            result += i # result = result+i
    elif operation == 'mul':
        result = 1
        for i in args:
            result *= i
    return result

def say_myself(name, age, dept="AI-Engineering"):
    print(f'나의 이름은 {name}')
    print(f'나이는 {age}')
    print(f'학과는 {dept}')

say_myself('전진아', 26)