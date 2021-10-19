from enum import Enum
from fixed_stack import FixedStack

Menu = Enum('Menu_list', ['push', 'pop', 'peek', 'Dump', 'Quit'])

def select_menu():
    s = [f'({m.value}){m.name}' for m in Menu]
    while True:
        print(*s, sep=' ', end='')
        n = int(input(': '))
        if 1 <= n <= len(Menu):
            return Menu(n)

s = FixedStack(5)

while True:
     print(f'현재 데이터 개수: {len(s)} / {s.capacity}')
     menu = select_menu()

     if menu == Menu.push:
         x = input("데이터를 입력하세요: ")
         try:
             s.push(x)
         except FixedStack.Full as e:
            print(e)
     elif menu == Menu.pop:
         try:
             s.pop()
         except FixedStack.Empty:
             print("스택이 비어 있습니다.")
