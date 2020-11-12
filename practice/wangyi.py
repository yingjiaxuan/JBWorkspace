import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = int(sys.stdin.readline().strip())
    ls = []
    d = {}
    for i in range(n):
        # 读取每一行
        line = sys.stdin.readline().strip()
        # 把每一行的数字分隔后转化成int列表
        ls.append(line.split())

    for i in ls:
        if i[1] in d:
            d[i[1]] += 1
        else:
            d[i[1]] = 1

    val = 0
    for key in d:
        if d[key] >=2 :
            val += 1

    print (val)
