import sys


def processor(pp_num, person_num, team_num):
    time = 40
    return time
    pass

def time_maker(time):
    hour = time/60
    hour = 8 + hour
    if hour > 12 :
        flag = 'pm'
        hour = hour -12
        return hour

if __name__ == "__main__":
    lst = []
    n = int(sys.stdin.readline().strip())
    lst.append(n)
    for i in range(n):  # 读取输入
        num = int(sys.stdin.readline().strip())
        lst.append(num)
        if num != 1:
            for j in range(2):
                line = sys.stdin.readline().strip()
                values = list(map(int, line.split()))
                lst.append(values)
        else:
            line = sys.stdin.readline().strip()
            values = list(map(int, line.split()))
            lst.append(values)
            lst.append(values)


    print(lst)
