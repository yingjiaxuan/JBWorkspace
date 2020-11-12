import sys
if __name__ == "__main__":
    # 读取第一行的n
    n = str(sys.stdin.readline().strip())
    n = n.split(' ',1)
    index = []
    m_i = []
    r_i = []
    for i in range(int(n[0])):#录入
        # 读取每一行
        line = str(sys.stdin.readline().strip())
        line = line.split(' ',1)
        index.append(line)
        m_i.append(line[0])
        r_i.append(line[1])

    #print(n,index,m_i,r_i)

