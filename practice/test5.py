# all_num = 0
# def minus (num):#初始输入10
#     global all_num
#     if num == 0:
#         all_num = all_num +1
#         return 0
#     if num == -1:
#         return 0
#     if num > 0:
#         minus (num-1)
#         minus (num-2)
#
# num = 10
# minus(num)
# print (all_num)


import sys


def odd_iter():
    n = 1
    while True:
        n = n + 2
        yield n


# 过滤掉n的倍数的数。
def not_divisible(n):
    return lambda x: x % n > 0


# 获取当前序列的第一个元素，然后删除后面序列该元素倍数的数，然后构造新序列。
def primes():
    yield 2
    it = odd_iter()
    while True:
        n = next(it)
        yield n
        it = filter(not_divisible(n), it)


def make_prime(stop):  # 构建素数表
    start = 1
    lst = []
    for n in primes():
        if start < n < stop:
            lst.append(n)
        elif n > stop:
            break
    return lst


all_num = 0


def sum(num):
    lst = make_prime(n)
    global all_num
    for i in range(len(lst)-1,0,-1):
        num = num - lst[i]
        print (lst[i])

    pass


if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    # 获取 start 到 stop 之间的素数。

    sum(n)

    print(all_num)

    pass
