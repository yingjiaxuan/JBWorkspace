import sys
import copy
all_num = 0
test_list = {'E':0,'M':0,'H':0}
def check_list (test_list):
    if test_list['E'] == 1 and test_list['M']==1 and test_list['H']==1:
        return 0 #表示组合成功
    else:
        return -1


def combo(a,a_nom):
    global all_num
    global test_list
    if check_list(test_list) == 0:
        all_num = all_num+1
        a = copy.deepcopy(a_nom)
        return
    if test_list["E"] == 0 and (int(a[0]) > 0 or int(a[1])>0):
        if a[0]>0:
            a_1 = copy.deepcopy(a)
            a_1[0] = a_1[0]-1
            test_list['E'] = 1
            combo(a_1,a_nom)
        if a[1]>0:
            a_1 = copy.deepcopy(a)
            a_1[1] = a_1[1]-1
            test_list['E'] = 1
            combo(a_1,a_nom)
    if test_list["M"] == 0 and (int(a[1]) > 0 or int(a[2])>0 or int(a[3])>0):
        if a[1]>0:
            a_1 = copy.deepcopy(a)
            a_1[1] = a_1[1]-1
            test_list['M'] = 1
            combo(a_1,a_nom)
        if a[2]>0:
            a_1 = copy.deepcopy(a)
            a_1[2] = a_1[2]-1
            test_list['M'] = 1
            combo(a_1,a_nom)
        if a[3]>0:
            a_1 = copy.deepcopy(a)
            a_1[3] = a_1[3]-1
            test_list['M'] = 1
            combo(a_1,a_nom)
    if test_list["H"] == 0 and (int(a[3]) > 0 or int(a[4])>0):
        if a[3]>0:
            a_1 = copy.deepcopy(a)
            a_1[3] = a_1[3]-1
            test_list['H'] = 1
            combo(a_1,a_nom)
        if a[4]>0:
            a_1 = copy.deepcopy(a)
            a_1[4] = a_1[4]-1
            test_list['H'] = 1
            combo(a_1,a_nom)


if __name__ == "__main__":
    for line in sys.stdin:
        a = line.split()
        break
    print(a)
    a = list(map(int,a))

    # a = [2,2,1,2,2]
    a_nom = copy.deepcopy(a)
    combo(a,a_nom)
    print (all_num)
