from __future__ import annotations
from typing import Any, Type
import hashlib

# 해시를 구성하는 Node Class
class Node:
    def __init__(self, key: Any, value: Any, next: Node) -> None:
        self.key = key
        self.value = value
        self.next = next

class ChainedHash:

    def __init__(self, capacity: int) -> int:
        self.capacity = capacity
        self.table = [None] * self.capacity

    def hash_value(self, key: Any) -> int:
        if isinstance(key, int):
            return key % self.capacity
        return (int(hashlib.sha256(str(key).encode()).hexdigest(),16)% self.capacity)

    # 찾고자 하는 키(값)에 Pair가 되는 Value를 반환하는 함수(method)
    def search(self, key: Any) -> Any:
        hash = self.hash_value(key)
        # node_ptr
        p = self.table[hash]

        while p is not None:
            if p.key == key:
                return p.value
            p = p.next

        return None

    def add(self, key: Any, value: Any) -> bool:
        hash = self.hash_value(key)
        p = self.table[hash]

        # add 하고자 하는 값이 기존 버킷 내에 있는지를 선형 탐색
        while p is not None:
            if p.key == key:
                return False
            p = p.next

        temp = Node(key, value, self.table[hash])
        self.table[hash] = temp

        return True

    def dump(self) -> None:
        for i in range(self.capacity):
            p = self.table[i] # 각 버킷에 있는 노드를 하나씩 가져오기
            print(i, end='')
            while p is not None:
                print(f' ->{p.key}({p.value})', end='')
                p = p.next  # 다음 노드를 계속
            print()
