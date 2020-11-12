import sys
if __name__ == "__main__":
    # 读取第一行的n
    l_1 = list(str(sys.stdin.readline().strip()))
    l_2 = list(str(sys.stdin.readline().strip()))
    dict_1 = {}
    dict_2 = {}
    print (l_1)
    print (l_2)
    for i in range(len(l_1)):
        if i in l_1:
            dict_1[l_1[i]] += 1
        else:
            dict_1[l_1[i]] = 1
    for i in range(len(l_2)):
        if i in l_2:
            dict_2[l_2[i]] += 1
        else:
            dict_2[l_2[i]] = 1
    print(dict_1)
    print(dict_2)
    if dict_1==dict_2 :
        print (int(True))
    else:
        print (int(False))