def read_file(file_name):

    rtn_list = []
    cnt = 0

    try:
        f = open(file_name, "r")

        while True:
            line = f.readline()
            if not line: break

            # map : 리스트의 요소를 지정된 함수(float으로 행 변환)로 처리해 주는 함수, 내장함수
            map(float, (line.rstrip("\n")).split("\t"))
            rtn_list.append(list(map(float, ((line.rstrip("\n")).split("\t")))))
            cnt += 1
        f.close()


        return rtn_list, cnt
    except FileNotFoundError as e:
        print(e)

if __name__ == '__main__':
    rtn_list = read_file("./score_mlr03.txt")
    print(rtn_list)
