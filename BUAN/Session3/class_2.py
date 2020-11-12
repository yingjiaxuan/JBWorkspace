import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
import numpy as np
from sklearn import metrics


def summary_corf(model_object):  # 利用alpha去进行迭代
    n_predictors = X.shape[1]
    model_coef = pd.DataFrame(model_object.coef_.reshape(1, n_predictors), columns=X.columns.values)
    model_coef['Intercept'] = model_object.intercept_  # 神秘模型，等会要看懂，线性方程的截距
    return model_coef.transpose()
    pass


def profit_calculation(model, x_value, y_value):  # 输出平均收益
    d_cutoff = 1 / 11
    decision = list(model.predict_proba(x_value)[:, 1] > d_cutoff)
    sum = 0
    for i in range(len(decision)):
        if decision[i] == True and y_value.iloc[i] == 1:
            profit = 10
        elif decision[i] == True and y_value.iloc[i] == 0:
            profit = -1
        else:
            profit = 0
        sum = sum + profit
    print(sum / len(decision))
    pass


def profit_calculation_usage(model, x_value, y_value):  # 输出平均收益
    d_cutoff = 1 / 11
    decision = list(model.predict_proba(x_value)[:, 1] > d_cutoff)
    y = list(y_value)
    sum = 0
    for i in range(len(decision)):
        if decision[i] == True and y[i] == 1:
            profit = 10
        elif decision[i] == True and y[i] == 0:
            profit = -1
        else:
            profit = 0
        sum = sum + profit
    average = sum / len(decision)
    return average
    pass


if __name__ == "__main__":
    '''
    - Classification rule becomes irrelevant
     Now decision rule matters
    - Confusion matrix becomes irrelevant
     Now decision matrix matters
    - Misclassification error rate/accuracy becomes irrelevant
     Now average profits matters
    
    So, we need a Decision Cut-Off
    
    
    Change confusion matrix into decision matrix. ——主要是因为给钱别人不一定会接，也就是和准确率不同的额外一层逻辑，使用这个去比较每
    个model的收益
    
    What is the minimum probability that you want to see to justify the decision to SEND?
    10*x + (-1)*(1-x) > 0 , so ,SEND is that probability > 1/11
    '''
    df = pd.read_csv('/Users/simon/JBWorkspace/BUAN/Session3/Ref_File/PersonalLoan.csv')
    print(df.head(10))
    print(df.columns.values)
    print(df.shape)
    print(df.isnull().sum())

    rvar_list = ['ZIPCode']
    df_sample1 = df.drop(columns=rvar_list)
    print(df_sample1)

    cvar_list = ['Education', 'SecuritiesAccount', 'CDAccount', 'Online', 'CreditCard', 'PersonalLoan']
    nvar_list = ['Age', 'Experience', 'Income', 'Family', 'CCAvg', 'Mortgage']

    df_sample2 = df_sample1.copy()
    df_sample2[nvar_list] = (df_sample1[nvar_list] - df_sample1[nvar_list].mean()) / df_sample1[nvar_list].std()

    print(df_sample2)

    df_sample3 = df_sample2.copy()
    df_sample3[cvar_list] = df_sample2[cvar_list].astype('category')
    df_sample3[nvar_list] = df_sample2[nvar_list].astype('float64')  # 引用的问题

    df_sample4 = df_sample3.copy()
    df_sample4 = pd.get_dummies(df_sample4, prefix_sep='_')
    print(df_sample4.columns.values)

    # rdummies = ['Education_1','SecuritiesAccount_No','CDAccount_No','Online_No','CreditCard_No','PersonalLoan_No']
    rdummies = ['Education_1', 'SecuritiesAccount_Yes', 'CDAccount_Yes', 'Online_Yes', 'CreditCard_Yes',
                'PersonalLoan_No']
    df_sample5 = df_sample4.copy()
    df_sample5 = df_sample5.drop(columns=rdummies)

    print(df_sample5.columns.values)
    print(df_sample5)

    # 以上完成Pre-processing
    df4partition = df_sample5.copy()
    testpart_size = 0.2
    df_nontestData, df_testData = train_test_split(df4partition, test_size=testpart_size, random_state=1)
    print('This is nontestData')
    print(df_nontestData)
    print('')
    print('This is testData')
    print(df_testData)

    # Logistic Regression with penalty
    DV = 'PersonalLoan_Yes'
    y = df_nontestData[DV]
    X = df_nontestData.drop(columns=[DV])

    alpha = 10
    clf = LogisticRegression(C=1 / alpha, penalty='l1', solver='saga', max_iter=200, random_state=1).fit(X, y)
    # 使用l1的penalty,saga类似于penalty type， max_iter是迭代次数, 同样目的是拿出一个合适的alpha

    print(summary_corf(clf))

    y_test_actual = df_testData[DV]
    print(y_test_actual)
    X_test = df_testData.drop(columns=[DV])
    print(X_test)

    outcome = clf.predict_proba(X_test)[:, 1]  # 输出预测的"概率"

    profit_calculation(clf, X_test, y_test_actual)

    kfolds = 5
    min_alpha = 0.01
    max_alpha = 100

    max_C = 1 / min_alpha
    min_C = 1 / max_alpha  # 设置迭代步长list

    n_candidates = 5000  # 完全没听到

    C_list = list(np.linspace(min_C, max_C, num=n_candidates))

    print(C_list)  # 0.001到100平均分1000个数字

    clf_optimal = LogisticRegressionCV(Cs=C_list, cv=kfolds, scoring=profit_calculation_usage, penalty='l1',
                                       solver='saga',
                                       max_iter=200, random_state=1, n_jobs=-1).fit(X,
                                                                                    y)  # 增加了一个scoring，代表找scoring输出的结果作为模型结果
    print(summary_corf(clf_optimal))
    print(1 / clf_optimal.C_)  # 整出alpha，也就是我们要的结果
    pass
