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

file=r"/Users/simon/JBWorkspace/BUAN/Session2/Ref_File/XYHW/HW3_Airfares_Selected.csv"
td=pd.read_csv(file)
cvar_list = ['VACATION', 'SW', 'SLOT', 'GATE']  # categorical
nvar_list = ['S_INCOME', 'E_INCOME', 'S_POP', 'E_POP', 'DISTANCE', 'FARE']  # numerical
td[nvar_list] = (td[nvar_list] - td[nvar_list].mean()) / td[nvar_list].std()

td[cvar_list] = td[cvar_list].astype('category')  # 强制类型转换
td[nvar_list] = td[nvar_list].astype('float64')

td = pd.get_dummies(td, prefix_sep='_') # 这种状态编码，3个情况会有100，010，001，但是我们其实只需要10，01，00
rdummies = ['VACATION_No', 'SW_No', 'SLOT_Free', 'GATE_Constrained']  # 去除多余dummy(状态编码)
td = td.drop(columns=rdummies)
pass
def summary_corf(model_object):  # 利用alpha去进行迭代
    n_predictors = X.shape[1]
    model_coef = pd.DataFrame(model_object.coef_.reshape(1, n_predictors), columns=X.columns.values)
    model_coef['Intercept'] = model_object.intercept_  # 神秘模型，等会要看懂
    return model_coef.transpose()
    

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
#print(clf_optimal)
print(clf_optimal.alpha_)

alpha = 0.1  # Choose a penalty level
alpha_1 = clf_optimal.alpha_  # 来自LassoCV的alpha
clf = Lasso(alpha=alpha_1, random_state=1).fit(X, y)  # X,predictors
print(clf)
model_object = clf

print("alpha = 0.1")
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

corr = td.corr()
#print (corr)

url = r"/Users/simon/JBWorkspace/BUAN/Session2/Ref_File/XYHW/newfile.csv"
td = pd.read_csv(url)
#print (td)
cvar_list = ['VACATION', 'SW', 'SLOT', 'GATE']  # categorical
nvar_list = ['S_INCOME', 'E_INCOME', 'S_POP', 'E_POP', 'DISTANCE', 'FARE']  # numerical
td[nvar_list] = (td[nvar_list] - td[nvar_list].mean()) / td[nvar_list].std()

td[cvar_list] = td[cvar_list].astype('category')  # 强制类型转换
td[nvar_list] = td[nvar_list].astype('float64')

td = pd.get_dummies(td, prefix_sep='_') # 这种状态编码，3个情况会有100，010，001，但是我们其实只需要10，01，00
rdummies = ['VACATION_No', 'SW_No', 'SLOT_Free', 'GATE_Constrained']  # 去除多余dummy(状态编码)
td = td.drop(columns=rdummies)
pass
predicted_standardizedPrice = clf_optimal.predict(td) # 套用之前的clf_optimal,出标准化结果
print (predicted_standardizedPrice)

   