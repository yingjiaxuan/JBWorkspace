import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

def fun_close_pic(min = 1):
    plt.draw()
    plt.pause(min)
    plt.close()

if __name__ == "__main__":
    # 1，2，3 three box plots
    file = "/Users/simon/JBWorkspace/BUAN/Session1/Ref_File/HW2_Cereals.csv"
    td = pd.read_csv(file)

    data_1 = td[['shelf','rating']]
    row, column = data_1.shape

    tem_1 = pd.DataFrame()
    tem_2 = pd.DataFrame()
    tem_3 = pd.DataFrame()
    for row_num in range(row): # int i = 1;for(i=1;i<=100;i++){}
        a = int(data_1.loc[row_num,['shelf']])
        if a == 1:
            ls = data_1.iloc[row_num]
            tem_1 = tem_1.append(ls)
        if a == 2:
            ls = data_1.iloc[row_num]
            tem_2 = tem_2.append(ls)
        if a == 3:
            ls = data_1.iloc[row_num]
            tem_3 = tem_3.append(ls)
        pass
    tem_1 = pd.DataFrame(np.array(tem_1[['rating']]),columns=list('1'))
    tem_2 = pd.DataFrame(np.array(tem_2[['rating']]),columns=list('2'))
    tem_3 = pd.DataFrame(np.array(tem_3[['rating']]),columns=list('3'))
    a = pd.concat([tem_1,tem_2,tem_3],axis=1)
    print(a)

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    # plt.xlabel (['1','2','3'])
    # plt.ylabel (['ppp'])
    plt.suptitle('Q1S1')
    ax = a.boxplot()
    plt.show()

    print (td)
    corr = td.corr()
    print (corr)
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()
    pass