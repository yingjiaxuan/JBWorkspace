import sys


class people(object):
    def __init__(self, W, H):
        self.H = H
        self.W = W
        self.bg = []
        self.P = 0
        self.Q = 0
        self.sp = []
        self.i = 0
        self.j = 0
        self.a = 0
        self.b = 0

    def get_bg(self, str):
        lst = list(str)
        self.bg.append(lst)

    def get_PQ(self, P, Q):
        self.P = P
        self.Q = Q

    def get_sp(self, str):
        lst = list(str)
        self.sp.append(lst)

    def get_ijab(self, i, j, a, b):
        self.i = i
        self.j = j
        self.a = a
        self.b = b

    def print_all(self):
        print(self.H,self.W)
        print(self.bg)
        print(self.P,self.Q)
        print(self.sp)
        print(self.i,self.j,self.a,self.b)

    def get_origin(self):
        pic = self.bg
        sp_location = []
        for i in range(self.Q):#获得实际小人每个点的坐标
            for j in range(self.P):
                location=[]
                location.append((self.i+i,self.j+j))
                sp_location.append(location)
        for i in sp_location:
            if i[0]>=1 and i[1]>=1:
                index = sp_location.index(i)
                pic[i[0]][i[1]] = self.sp[index/self.Q][index % self.P]
            pass
    pass

def change_pic(pic,a,b):
    pass

def check_end():
    pass

if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ls = []
    for i in range(n):#录入每一组数据
        # 读取每一行
        line = sys.stdin.readline().strip()
        values = list(map(int, line.split()))
        act = people(values[0], values[1])

        for j in range(values[1]):  # 录入背景
            line = sys.stdin.readline().strip()
            act.get_bg(line)

        line = sys.stdin.readline().strip()
        values = list(map(int, line.split()))
        act.get_PQ(values[0], values[1])

        for j in range(values[1]):  # 录入图像
            line = sys.stdin.readline().strip()
            act.get_sp(line)

        line = sys.stdin.readline().strip()
        values = list(map(int, line.split()))
        act.get_ijab(values[0], values[1], values[2], values[3])

        ls.append(act)
        continue

    ls[0].print_all()

    # for i in range(n):
    #     origin_pic = get_origin()
    #     while (check_end()):
    #         pic_1 = origin_pic
    #         pic_2 =
    #         pass
