import pandas as pd
import time


def print_time():
    global time_1
    tem_t = time.time()
    tem = tem_t - time_1
    z = round(tem, 3)
    print("累计消耗时间为:" + str(z))


time_1 = time.time()

if __name__ == "__main__":
    # 读取

    train_data = '/Users/simon/OpenSource/2020年C题/附件1-P300脑机接口数据/S1/S1_train_data.xlsx'  # train_data路径
    train_event = '/Users/simon/OpenSource/2020年C题/附件1-P300脑机接口数据/S1/S1_train_event.xlsx'  # train_event路径
    td = pd.read_excel(train_data, sheet_name="char01(B)",header=None)  # 表格名字，处理的时候记得修改
    print (td)
    row_td, column_td = td.shape
    te = pd.read_excel(train_event, sheet_name="char01(B)",header=None)  # 表格名字，处理的时候记得修改
    row_te, column_te = te.shape
    print (te)

    res = pd.DataFrame()
    for row_loop in range(row_te):  # 循环test event表
        t_1 = int(te.iloc[row_loop, 1]) - 1
        ls = td.iloc[t_1:(t_1 + 1)]  # 取出对应
        res = res.append(ls)

    print(res)
    print_time()
    # 录入
    t = "/Users/simon/OpenSource/2020年C题/data.xlsx"  # 改成目标的excel路径
    res.to_excel(t, sheet_name='char01(B)', index=False, header=True)
