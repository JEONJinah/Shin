class FixedStack:

    class Empty(Exception):
        pass

    class Full(Exception):
        def __str__(self):
            return "스택이 가득 찼습니다."

    # stk1 = FixedStack()
    def __init__(self, capacity=64):
        # stk: (stk 객체 변수) 스택 배열? (리스트)
        self.stk = [None] * capacity  # stk1. stk = [None] * capacity
        self.capacity = capacity
        self.ptr = 0

    def __len__(self):
         return self.ptr


    def is_empty(self):
        return self.ptr <= 0

    def is_full(self):
        rtn_val = False
        if self.ptr >= self.capacity:
            rtn_val = True
        return  rtn_val  # return self.ptr >= self.capacity

    def push(self, value):
        if self.is_full():
            raise FixedStack.Full
        self.stk[self.ptr] = value
        self.ptr += 1

    def pop(self):
        if self.is_empty():
            raise FixedStack.Empty
        self.ptr -= 1
        return self.stk[self.ptr]

    def peek(self):
        if self.is_empty():
            raise FixedStack.Empty
        return self.stk[self.ptr - 1]
    def clear(self):
        self.ptr = 0

    def find(self, value):
        for i in range(self.ptr -1, -1, -1):
            if self.stk[i] == value:
                return i
        return -1

    def dump(self):
        if self.is_empty():
            raise FixedStack.Empty
        else:
            print(self.stk[:self.ptr])

