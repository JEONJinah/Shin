heat_tree = 0
while heat_tree < 10:
    heat_tree += 1   # heat_tree = heat_tree + 1 , [heat_tree++]
    print(f"나무를 {heat_tree}번 찍었습니다.")
    if heat_tree == 10:
        print("나무 넘어감")

prompt = """   
    1. ADD
    2. DEL
    3. QUIT
    Enter Number:
"""           # 빠져나갈 때까지 치는 코드

number = 0
while number < 3:
    print(prompt)
    number = int(input())

# 커피 자판기

coffee = 3
while True:
    money = int(input("insert money: "))
    if not coffee:
        print("커피가 없습니다")
        break
    if money == 300:
        print("커피 출력")
        coffee -= 1
    elif money > 300:
        print(f'{money-300}원 반환 + 커피 출력')
        coffee -= 1
    else:
        print(f'돈이 모자랍니다')