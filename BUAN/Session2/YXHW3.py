import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def column_nuique_values(x_column):
    return list(x_column.unique())
    pass


def summary_corf(model_object):  # 利用alpha去进行迭代
    n_predictors = X.shape[1]
    model_coef = pd.DataFrame(model_object.coef_.reshape(1, n_predictors), columns=X.columns.values)
    model_coef['Intercept'] = model_object.intercept_  # 神秘模型，等会要看懂
    return model_coef.transpose()
    pass


if __name__ == "__main__":
    # 显示所有列
    pd.set_option('display.max_columns', None)
    # #显示所有行
    # pd.set_option('display.max_rows', None)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)

    file = r"/Users/simon/JBWorkspace/BUAN/Session2/Ref_File/XYHW/HW3_Airfares_Selected.csv"
    td = pd.read_csv(file)
    cvar_list = ['VACATION', 'SW', 'SLOT', 'GATE']  # categorical
    nvar_list = ['S_INCOME', 'E_INCOME', 'S_POP', 'E_POP', 'DISTANCE', 'FARE']  # numerical
    td_mean = td[nvar_list].mean()
    td_std = td[nvar_list].std()
    # td[nvar_list] = (td[nvar_list] - td[nvar_list].mean()) / td[nvar_list].std()
    td[nvar_list] = (td[nvar_list] - td_mean) / td_std

    td[cvar_list] = td[cvar_list].astype('category')  # 强制类型转换
    td[nvar_list] = td[nvar_list].astype('float64')

    td = pd.get_dummies(td, prefix_sep='_')  # 这种状态编码，3个情况会有100，010，001，但是我们其实只需要10，01，00

    rdummies = ['VACATION_No', 'SW_No', 'SLOT_Free', 'GATE_Constrained']  # 去除多余dummy(状态编码)

    print("去除dummy前")
    print(td)
    td = td.drop(columns=rdummies)

    td4partation = td.copy()
    testpart_size = 0.2
    td_nontestData, td_testData = train_test_split(td4partation, test_size=testpart_size,
                                                   random_state=1)  # 记得查一下random_state

    td_trainpartation = td_nontestData.copy()
    DV = 'FARE'
    y = td_trainpartation[DV]
    X = td_trainpartation.drop(columns=[DV])

    kfolds = 3  # 数据等分数，分别作为train和验证
    clf_optimal = LassoCV(cv=kfolds, random_state=1, n_jobs=-1).fit(X, y)  # 寻找最合适的alpha， Lasso根据你给的alpha来
    # print(clf_optimal)
    print(clf_optimal.alpha_)

    alpha = 0.1  # Choose a penalty level
    alpha_1 = clf_optimal.alpha_  # 来自LassoCV的alpha
    clf = Lasso(alpha=alpha_1, random_state=1).fit(X, y)  # X,predictors
    print(clf)
    model_object = clf

    print(summary_corf(model_object))
    print("xxxxx")
    print(summary_corf(clf_optimal))

    # 计算ASE
    y_test_actual = td_testData[DV]
    X_test = td_testData.drop(columns=[DV])
    y_test_predicted = clf.predict(X_test)
    n_obs_test = td_testData.shape[0]
    ASE_test = (sum((y_test_actual - y_test_predicted) ** 2)) / n_obs_test
    print(ASE_test)  # How much deviation is OK?
    pass

    corr = td.corr()
    # print (corr)
    # (h)

    test_dict = {'VACATION': ['No'], 'SW': ['No'], 'S_INCOME': [28760], 'E_INCOME': [27664], 'S_POP': [4557004],
                 'E_POP': [3195503],
                 'SLOT': ['Free'], 'GATE': ['Free'], 'DISTANCE': [1976]}
    cvar_list = ['VACATION', 'SW', 'SLOT', 'GATE']  # categorical
    nvar_list = ['S_INCOME', 'E_INCOME', 'S_POP', 'E_POP', 'DISTANCE', 'FARE']  # numerical

    td_sample = pd.DataFrame(test_dict)  # 计算用数据

    var_list = nvar_list.copy()
    var_list.remove('FARE')

    td_sample2 = td_sample.copy()
    td_sample2[cvar_list] = td_sample[cvar_list].astype('category')
    td_sample2[var_list] = td_sample[var_list].astype('float64')  # 强制类型转换

    historical_sample_mean = td_mean
    historical_sample_std = td_std

    td_sample2[var_list] = (td_sample2[var_list] - historical_sample_mean[var_list]) / historical_sample_std[
        var_list]  # 标准化

    td_sample2 = pd.get_dummies(td_sample2, prefix_sep='_')

    td_sample3 = td_sample2.copy()
    td_sample3.loc[0, 'VACATION_No'] = 0
    td_sample3.loc[0, 'SW_No'] = 0
    td_sample3.loc[0, 'SLOT_Free'] = 0
    td_sample3 = td_sample3.rename(
        columns={'VACATION_No': 'VACATION_Yes', 'SW_No': 'SW_Yes', 'SLOT_Free': 'SLOT_Controlled'
                 })  # 修改符合模型的数据源
    print(td_sample3.columns.values)

    predicted = clf_optimal.predict(td_sample3)

    predicted_value = predicted * historical_sample_std['FARE'] + historical_sample_mean['FARE']
    print(predicted_value) #最终结果

    # S_INCOME         0.019475
    # E_INCOME         0.062388
    # S_POP            0.030709
    # E_POP            0.049143
    # DISTANCE         0.605753
    # VACATION_Yes    -0.614716  Yes是0，No是1（No的情况）
    # SW_Yes          -0.641363  Yes是0 , (NO)
    # SLOT_Controlled  0.132911  Controlled是0 （Free）
    # GATE_Free       -0.309793  Gate是0 ，
    # Intercept        0.564667
