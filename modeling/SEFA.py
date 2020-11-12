import pandas as pd
import numpy as np
if __name__ == "__main__":
    file = "/Users/simon/Downloads/qq下载/UsedCar_MissingValue(1).csv"
    td = pd.read_csv(file)
    a = td.std()
    print(a)
    print(a.max())
    print(a.idxmax())