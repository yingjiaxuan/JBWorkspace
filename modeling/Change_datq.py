import pandas as pd
import numpy as np


def fun_change_data(url, sn):
    source_data = url
    sd = pd.read_excel(source_data, sheet_name=sn, header=None)
    # print(sd)
    out_list = []
    for row_loop in [x * 40 for x in range(2400 // 40)]:
        pd_tem = sd.iloc[row_loop:(row_loop + 40)]  # 一次取40行，一行20个属性，也就是40*20
        # print(pd_tem)
        ls_tem = pd_tem.values.tolist()  # 40 * 20
        for i in range(40):  # 此处我把40写死了
            ls_tem[i] = np.array(ls_tem[i])

        out_list.append([np.array(ls_tem)])  # 1 * 40 * 20循环2400//40次

    out_list = np.array(out_list)  # 最终结果
    print("最终输出向量组为：")
    print(out_list)  # 60*1*40*20
    print("每一层元素个数依次为: %d %d %d %d" % (len(out_list), len(out_list[0]), len(out_list[0][0]), \
                                       len(out_list[0][0][0])))

    return out_list
    pass


def fun_change_file(url, sn, url2):
    source_data = url
    sd = pd.read_excel(source_data, sheet_name=sn, header=None)
    row, column = sd.shape
    print(row, column)
    if row / 44 * 44 != row:  # 去尾
        tem = row / 44 * 44
        sd = sd.iloc[0:tem]
    row, column = sd.shape
    res = pd.DataFrame()
    for row_loop in [x * 44 for x in range(row // 44)]:
        pd_tem = sd.iloc[row_loop:(row_loop + 44)]
        pd_tem = pd_tem.iloc[0:40]
        res = res.append(pd_tem)
    print(res)
    res.to_excel(url2, sheet_name=sn, index=False, header=False)

    return res


# 以下为主函数，为操作演示，其中函数输入两个参数依次为，文件路径，excel子表名称
if __name__ == "__main__":  # 60*1*40*20
    source_url = '/Users/simon/OpenSource/2020年C题/data.xlsx'
    sheet_name = 'char01(B)'
    goal_url = ''  # 修改函数目标文件路径
    out_data = fun_change_data(source_url, sheet_name)  # 使用时，复制上方fun_change_data的全部代码，仿照以上三行即可
    out_file = fun_change_file(source_url, sheet_name, goal_url)  # 输出到文件
