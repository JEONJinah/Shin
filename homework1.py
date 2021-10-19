# 3번  Slide 13 basic knowledge   simple.py
languages = ['python', 'perl', 'c ', 'java']

for lang in languages:  # language 리스트의 lang 변수 동안
    if lang in ['python', 'perl']:  # lang 변수가 python 또는 perl이라면
        print("%6s need interpreter" % lang)  # lang 변수는 6자리를 차지하며, need interpreter 라는 문장을 이어서 출력해라.
    elif lang in ['c', 'java']:    # lang 변수가 c 또는 java 라면
        print("%6s need compiler" % lang)     # lang 변수는 6자리를 차지하며, need compiler라는 문장을 이어서 출력하라
    else:
        print("should not reach here")     # 여기까지 오면 안됨

# 9번 Floating point

print(f"{3.42134234:0.4f}")  # 소수점 4자리를 제외하고는 삭제
str2 = f"{3.42134234:10.3f}"
print(str2, len(str2))    # 앞에 빈칸 부터 3.421까지는 총 10자리이고, 소수점 3개만 남긴다.


# 10번 소수판단 여부
testNum = 9
for i in range(2, testNum):
    a = testNum % i
    if a == 0:
        break
if a == 0:
    print(f'{testNum} is not prime Number')
else:
    print(f'{testNum} is Prime Number')

# 11번 피싱코드 설명
# label data
# class_name position_x position_y width height
# width가 100 이상인 경우 class_name이 vehicle인 것을 Truck으로 업데이트

str_txt = "vehicle 10 10 50 50 vehicle 50 50 250 250 vehicle 100 100 10 100"

sp = str_txt.split()
print(sp)

j = 0
i = sp.count("vehicle")
idx1 = sp.index("vehicle")

while j < i:
    idx = idx1 + 5 * j
    j += 1
    print(sp[idx + 3])
    width_value = sp[idx + 3]
    if int(width_value) >= 100:
        sp[idx] = "Truck"
print(sp)