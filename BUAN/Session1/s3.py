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
    file = "/Users/simon/JBWorkspace/BUAN/Session1/Ref_File/HW2_Airfares.csv"
    td = pd.read_csv(file)
    corr = td.corr()
    print(corr)
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.show()
    pass