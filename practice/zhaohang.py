# a = '1232....flow_cts_source_group_tag'
#
# b = a.replace('..','.').replace('..','.').replace('.','\':\'')
#
# c = b.split('\n')
# for line in c:
#     line = '\''+line+'\','
#     print(line)

import sys

def jump(start,end,lst,zero_ls):
    lst = lst[start:end]
    for i in range(len(lst)):
        t = -i-1
        if lst[t] > i+1:
            return True
    return False
    pass

def get_zero(lst):
    re_list = []
    for i in range(len(lst)):
        if lst[i] == 0:
            re_list.append(i)
    return re_list

# def check_jump(start,end,lst,zero_ls):
#     ck_ls = []
#     for i in zero_ls:
#         if i <= start - 1:
#             ck_ls.append(i)
#     if ck_ls == []:
#         status = False
#
#     pass

def process(lst):
    zero_ls = get_zero(lst)
    if 0 not in lst:
        return True
    elif lst[0] == 0:
        return False
    else:
        flag = 0
        for i in range(len(lst)):
            if lst[i] == 0:
                status = jump(flag,i,lst,zero_ls)
                flag = i+1
                # if status == False:
                #     status = check_jump(flag,i,lst,zero_ls)
                return status



        return True

if __name__ == "__main__":
    # 读取第一行的n
    n = sys.stdin.readline().strip()
    n = n.split('[')
    n = n[1].split(']')
    values = list(n[0].split(','))
    # print(values)
    values = list(map(int,values))
    res = process(values)
    if res == False:
        print('false')
    else:
        print('true')
    print (get_zero(values))