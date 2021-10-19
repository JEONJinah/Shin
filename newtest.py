# 코드 작성 시 여러가지 방법으로 생각해보고 규현해볼 것

# 2500보다 작은 자연수 중에서 3 또는 7의 배수를 모두 더한 결과 값 구하기
total = 0
for i in range(2500):
    if i % 3 == 0:
        total += i
    elif i % 7 == 0:
        total += i
    elif i % 3 == 0 and i % 7 == 0:
        total -= i
print(total)

# 1-2
result = 0
for n in range(1, 2500):
    if n % 3 == 0 or n % 7 == 0:
        result += n
print(result)

# 1-3
sum(range(0, 2500, 3)) + sum(range(0, 2500, 7)) - sum(range(0, 2500, 21))
print(sum({i for j in (3, 7) for i in range(0, 2500, j)}))


# 피보나치 수열에서 4백만 이하이면서 짝수인 항의 합
a = 0
b = 1
c = a + b
total = 0

while c < 4000000:
    c = a + b
    a = b
    b = c
    if c % 2 == 0:
        total += c
print(total)

# 2-2

import time

count_t = time.time()
i = 1
j = 2
total = 2
while True:
    k = i + j
    if k % 2 == 0:
        total += k
    if k > 4000000:
        break;
    i = j
    j = k

print("{:d}".format(total))
