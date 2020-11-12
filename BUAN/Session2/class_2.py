import pandas as pd
import numpy as np

if __name__ == "__main__":
    '''
    Traditional way of running a linear regression: OLS method (ordinary least squares) 最小二乘法
    Model deployment --- Score new data
    So, should we plug the original data of new file into the selected model directly?
    No, we should standardize data. (值-均值)/标准差
    Of course, we should convert outcome into original data.
    
    Four questions we have answered
    - What is data partition?
    - When do we need to partition data?
    - Why do we need to partition data?
    - How do we partition data in Python?
    
    If we use only 60% percent of data to be training data, is it waste?
    Overfitting
    Data partition can help detect the overfitting issue
    - If a model overfits, we can detect the overfitting through its relative poor predictive performance over the test 
    partition. Specifically, we can compare
    1. The 'predictive performance' of the model over the non-test partition
    2. The 'predictive performance' of the model over the test partition
    '''
    # Part 7 Score the new data
    test_dict = {'Vacation':['No'] , 'SW':['No'], 'S_INCOME':[28760],'E_INCOME':[27664],'S_POP':[4557004],'E_POP':[3195503],
                 'SLOT':['Free'],'GATE':['Free'],'DISTANCE':[1976]}
    td = pd.DataFrame(test_dict)
    print(td)

    pass
