# 쉘 정렬은 h에 따른 phase 가 존재할 뿐이고, 구현은 단순 삽입정렬과 거의 유사함
# 0~ n-1을 1~n 번째까지 하나씩 옮기는 것이 아니라 h 값만큼 띄어가면서 삽입.
# 쉘 정렬: 정해진 h마다 그룹으로 정렬을 수행(각각의 단계를 h-정렬이라고도 함)

def shell_sorting(a):
    # 입력받은 리스트의 길이를 구해서 for 문에 사용
    n = len(a)
    # h 값 셋팅 (입력데이터 //2: textbook example: number of list 8 ==> 4, 2, 1 h 정렬)
    h = n // 2

    # shell 정렬이 h 정렬의 조합 h 정렬을 모두 수행 후, h=1인 정렬은 단순 삽입정렬
    while h > 0:
        """
        textbook example: 8 1 4 2 7 6 3 5 
        h == 4, i, j, tmp 어떤 값이 들어가는지 코드로 확인
        i = 4, 5, 6, 7
        j = 0, 1, 2, 3
        tmp = 7 6 3 5 (Group(cmp/sorting): 8 1 4 2)
        (8, 7), (1, 6), (4, 3), (2, 5) 
        ==> 7,8 1,6 3,4
        """

        for i in range(h, n):
            j = i - h
            tmp = a[i]
            # j >= 0 인 이유는?  (그룹을 만들기 위해,
            # 단순 삽입정렬과 차이점: 단순 삽입정렬의 경우에는 0번째 인덱스가 정렬이 되어있다고 가정)
            # e.g.) h 정렬의 첫번째 (8, 7)
            # i = 4(value:8), j=0(value:7)
            # j ==> -4
            while j >= 0 and a[j] > tmp:
                a[j + h] = a[j]
                j -= h
                # 첫번째 a[-4+4] tmp 값 삽입 (7)
            a[j+h] = tmp
        h //= 2

if __name__ == "__main__":
    in_list = [8, 1, 4, 2, 7, 6, 3, 5]
    shell_sorting(in_list)
    print(in_list)