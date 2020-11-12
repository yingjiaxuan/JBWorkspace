import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def fun_close_pic(min = 1):
    plt.draw()
    plt.pause(min)
    plt.close()

if __name__ == "__main__":
    # p1
    df = pd.read_csv(r'/Users/simon/JBWorkspace/BUAN/Session1/Ref_File/UsedCar_MissingValue(1)(1).csv')
    # df = pd.read_csv('C:\Users\xiaoyu\AppData\Local\Programs\Python\Python38-32/UsedCar_MissingValue(1).csv')
    # print(df.head(10))
    # print(df.columns.values)
    # print(df.shape)

    #p2
    x_var = 'KM'
    y_var = 'Price'
    df.plot.scatter(x = x_var ,y = y_var,legend = False)
    fun_close_pic()

    df4heatmap = df
    corr = df4heatmap.corr()
    print(corr)
    print(corr.columns)
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)  # abs [0.1,0.5] possible， abs[0.5,0.9]
                                                                            # relevant, abs>0.9 mistake
    fun_close_pic()

    x_var = ['KM','CC']
    y_var = ['Price']
    df4ssb = df
    print(df4ssb)
    # ax = df4ssb.boxplot(column = y_var, by = x_var)
    ax = df4ssb.boxplot()
    # ax.set_ylabe(y_var)
    fun_close_pic()