import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
if __name__ == "__main__":
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    np.random.seed(19680801)
    fig = plt.figure()
    ax = plt.subplot()
    data = np.random.rand(50) * 10
    ax.set_title("aaa")
    ax.boxplot(data)
    plt.show()
    pass