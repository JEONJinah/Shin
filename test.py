a = [1, 2, 3]
b = a
c = a[:]
a.remove(1)
print(f'b={b}, c={c}')

price_list = [32100, 32150, 32000, 32500]
for i in range(len(price_list)):
    print(price_list[len(price_list)-(i+1)], end=" ")

str_t = "Hello"
idx = 1
while str_t:
    print(len(str_t), str_t)
    str_t = str_t[:len(str_t) - idx]