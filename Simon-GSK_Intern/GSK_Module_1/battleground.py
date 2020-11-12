import random

list = [1,2,3,4,5,6,7]  # +1+1，+3atk, +3lif, poison, Divine Shield, Windfury，Death
print(list)
set_evolve = input("Enter your evolve number: ")
i = 1
number = []

All = 0
for i in range (0,10000):
    evolve = 1
    number = []
    for evolve in range(0, int(set_evolve)):
        # 3.生成随机数
        num = random.randint(1, 7)
        # 4.添加到列表中
        number.append(num)
    if 4 in number and 5 in number:
        if 7 in number:
            pass
        else:
            All = All + 1
            print(number)

print (All/10000)
    # print(number)


# while i <= int(set_evolve):
#
#     pass