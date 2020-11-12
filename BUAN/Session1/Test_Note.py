import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def column_nuique_values(x_column):
    return list(x_column.unique())
    pass


if __name__ == "__main__":
    # P1 Data Pre-Precessing
    '''
    variable, predictor, observation
    1.Remove all records with missing values --- removed too much data
    2.If too many records with missing values
    So how many is 'noe too many'? Rule of thumb: less than 10%, no more than 20%
    
    2.1 Use institutional knowledge to recover (depend on data constraint)
    2.2 For numerical variables
        - Replace with the median value
        For categorical variables
        - Replace with mode --- the most frequent data
    2.3 On the dependent variable (因变量)
        Remove
        
    3. A special type of missing ---- Having value but uncertain --- use box plots --- 离谱数据
    
    Can it be used to predict?
    Not for regression methods. (回归模型) Some data cannot be used in linear model. (得他妈的是数字)
    方法——状态编码 01，10，11，00 —— categorical variable
    
    4. Numerical variable standardization  p'=（值-平均值）/标准差  用于处理类似于货币这种浮动量 —— Standardized Prices(去特征)
    
    Total：
    No missing values.
    All numerical values are within proper ranges.
    All are numerical values. 
    No obvious outliers.
    '''

    # p2 Using PY to do data pre-processing
    # pandas can fill missing data
    file = "/Users/simon/JBWorkspace/BUAN/Session1/Ref_File/UsedCar_MissingValue(1)(1).csv"
    td = pd.read_csv(file)
    print(td)
    # Part 3 Missing Value Imputation
    print(td.isnull())  # 输出TF，表明哪里missing
    print(td.isnull().sum())  # 输出空格数量（按照column）
    print(td.dropna())  # 输出不带na的
    print(td.dropna(subset=['Price', 'Doors']))  # subset参数为仅关注对应column的na
    df_sample = td.dropna(subset=['Price'])

    # Specify the variable types based on their definitions
    cvar_list = ['Fuel_Type', 'Metallic', 'Automatic', 'Doors']  # categorical
    nvar_list = ['Price', 'Age', 'KM', 'HP', 'CC', 'Quarterly_Tax', 'Weight']  # numerical

    print(df_sample[nvar_list].median())  # 输出对应列均值
    a = df_sample[nvar_list].median()
    df_sample2 = df_sample.copy()  # 深拷贝
    df_sample = df_sample[nvar_list].fillna(value=a)  # 填上（替代空格）,对所有的连续性变量作操作

    print(df_sample)

    print(df_sample2)
    print(df_sample2['Fuel_Type'].unique())  # 处理categorical，输出特殊值(只能对一列作操作，not a dataframe)
    # df_sample2[cvar_list].apply(pd.unique())
    print()
    print(df_sample2[cvar_list].apply(column_nuique_values))  # 批处理，查看总共有几种离散值

    # Now we neet to know which one is irregular
    irregular_var = 'Fuel_Type'
    irregular_value = '?'
    print(df_sample2[irregular_var].replace(irregular_value, np.nan))  # 替换，？和nan

    df_sample3 = df_sample2.copy()
    df_sample3[irregular_var].replace(irregular_value, np.nan)

    df_sample4 = df_sample3.copy()
    df_sample4[cvar_list] = df_sample3[cvar_list].fillna(value=df_sample3[cvar_list].mode().loc[0, :])  # 所有categorical\
    # 空格塞上众数,记得最后检查一下为啥要从0开始
    print(df_sample3[cvar_list].mode())  # 众数
    print(df_sample4)

    # Part 4 Variable transformation
    print(df_sample4['Price'].std())  # 输出标准差
    std = df_sample4.std()
    mean = df_sample4.mean()  # 平均值,类型是series
    print(std, mean)

    df_sample5 = df_sample4.copy()
    df_sample5[nvar_list] = (df_sample5[nvar_list] - df_sample5[nvar_list].mean()) / df_sample5[nvar_list].std()
    print(df_sample5[nvar_list])

    df_sample6 = df_sample5.copy()
    df_sample6[cvar_list] = df_sample5[cvar_list].astype('category')  # 强制类型转换
    df_sample6[nvar_list] = df_sample5[nvar_list].astype('float64')

    df_sample7 = pd.get_dummies(df_sample6, prefix_sep='_') # 这种状态编码，3个情况会有100，010，001，但是我们其实只需要10，01，00
    print(df_sample7)
    print(df_sample7.columns.values)
    rdummies = ['Fuel_Type_Petrol', 'Metallic_No', 'Automatic_N', 'Doors_5.0']  # 去除多余dummy(状态编码)
    df_sample8 = df_sample7.drop(columns=rdummies)
    print(df_sample8.columns.values)
    print (df_sample8)
    pass
