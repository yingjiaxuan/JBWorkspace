import numpy as np
import pandas as pd


if __name__ == "__main__":
    re_predicted = np.array([[1.111],[2.222],[3.333]])
    print(re_predicted)
    df = pd.DataFrame(re_predicted)
    print(df)

    t = 'url'  # 改成目标的excel路径
    df.to_excel(t, sheet_name='char01(B)', index=False, header=False)
    pass