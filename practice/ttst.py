import sys


class text:
    def __init__(self):
        self.car = []
        self.num_water = 0
        self.water_list = []

    def print_data(self):
        print(self.car)
        print(self.num_water)
        print(self.water_list)


def print_input(lst):
    for i in range(len(lst)):
        lst[i].print_data()
    pass


def water_processor(l1, l2, water):
    vertex = water[0]
    return 0
    pass


def check_water(text_o):
    line_1 = text_o.car[0]
    line_2 = text_o.car[2]
    point_num = 0
    point_list = []
    for i in range(text_o.num_water):
        point_1 = water_processor(line_1,line_2,text_o.water_list[i])

    pass


if __name__ == "__main__":
    T = int(sys.stdin.readline().strip())
    Data_input = []
    for i in range(T):
        data = text()
        line = sys.stdin.readline().strip()
        ADV = list(map(int, line.split()))  # 录入车车
        data.car = ADV
        n = int(sys.stdin.readline().strip())
        data.num_water = n
        for j in range(n):
            line = sys.stdin.readline().strip()
            water = list(map(int, line.split()))
            data.water_list.append(water)
        Data_input.append(data)
    for i in range(len(Data_input)):
        check_water(Data_input[i])
    pass
