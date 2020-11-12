import sys

class text:
    def __init__(self):
        self.car = []
        self.num_water = 0
        self.water_list = []

if __name__ == "__main__":
    T = int(sys.stdin.redline().strip())
    Data_input = []
    for i in range(T):
        data = text()
        line = sys.stdin.readline().strip()
        ADV = list(map(int, line.split())) # 录入车车
        Data_input.car = ADV
        n = int(sys.stdin.redline().strip())
        Data_input.num_water = n
        for j in range(n):
            line = sys.stdin.readline().strip()
            water = sys.stdin.readline().strip()
            Data_input.water_list.append(water)
    print(Data_input)
    pass