# if 문
sum_var = 0   #0은 FALSE를 나타냄

if not sum_var:   #전체 조건은 1 (즉, true여서 밑에 프린트 문 나옴)
    print("if not condition test!")

money = 2000
card = True
if money >= 3000 or card: #돈이 3000원 이상 있거나 카드가 있으면 택시타라
    print("Take Taxi")
else:
    pass

list_var = [1, 2, 3, 4, 5]
if 8 in list_var:
    print("8 is in the list")
else:
    print("Cant find number 8")

# 조건부 표현식
score = 50
message = "Success" if score >= 60 else "Failure"
print(message)
