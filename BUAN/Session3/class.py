# 9 Steps of building a prediction modle

# scatterplot 散点图
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


if __name__ == "__main__":
    '''
    1. Understand your purpose Identify DIDA.  Data, Insights, Decision, Advantage
    - Identify DIDA
    A customer's demographic data, mortgage, etc 客户的统计数据、抵押贷款等
    2. Understand your task
    - Check the type of the prediction in your DIDA
    . 
    3. Data acquisition（数据采集）
    - Individual level data with historical values of the dependent variable. 含因变量历史值的个体水平数据
    - Relevant and exante predictors 相关预测因子
    - Sufficient number of observations (the portrait shape) 足够数量的observations
     - If the dependent variable is binary, we need observations for both cases. 如果是二值的，则需要观察两种情况
     
    Expectation n = 10*u
    Probability n = 6*m*(u+1)  m是dependant data的量（yes or no就是2）,u是predictor的数量（12列属性）
    
    4. Data exploration
    5. Data pre-processing
    6. Selection of predictive techniques
    
    We should find a non-linear function as the prediction model
    PersonLoasn -> PR_ACCEPTANCE (化为概率)
    A new model's function form: logistic function.
    Cut-off value = 0.5 （逻辑回归的分类数据） p = 1/(1+e^-(b0+b1*INCOME+b2*……))
    
    So we also need to find the best logistic model. (Chosen of predictors)
    - High predictive performance
    - Parsimonious structure
     Less sophisticated. (Fewer predictors)
     
    How would we compare the predicted values(probabilities) with the actual values?
    - Convert binary to probability
    Classification
    因为用0.5划线会有误差，所以我们引入
    Confusion matrix
                    Predicted Class
    Actual Class    1       0
    1               201     85
    0               25      2689
    We can get accuracy rate from this. So we can ues this to compare different variables chosen.
    Add penalty level!!!
    We use error rate replace the ASE
    
    Conclusion: The model maximizes the accuracy is not necessary the model that maximizes the profit
    比如一个朴素模型，一个贷款都不发，正确率高于99%，但是收益是0
    '''
    # Class to show how to use logistic regression
    df = pd.read_csv('/Users/simon/JBWorkspace/BUAN/Session3/Ref_File/PersonalLoan.csv')
    print (df.head(10))
    print (df.columns.values)
    print (df.shape)
    print(df.isnull().sum())

    rvar_list = ['ZIPCode']
    df_sample1 = df.drop(columns=rvar_list)
    print(df_sample1)

    cvar_list = ['Education','SecuritiesAccount','CDAccount','Online','CreditCard','PersonalLoan']
    nvar_list = ['Age','Experience','Income','Family','CCAvg','Mortgage']

    df_sample2 = df_sample1.copy()
    df_sample2[nvar_list] = (df_sample1[nvar_list]-df_sample1[nvar_list].mean())/df_sample1[nvar_list].std()

    print(df_sample2)

    df_sample3 = df_sample2.copy()
    df_sample3[cvar_list] = df_sample2[cvar_list].astype('category')
    df_sample3[nvar_list] = df_sample2[nvar_list].astype('float64') #引用的问题

    df_sample4 = df_sample3.copy()
    df_sample4 = pd.get_dummies(df_sample3,prefix_sep='_')
    print(df_sample4.columns.values)

    rdummies = ['Education_1','SecuritiesAccount_No','CDAccount_No','Online_No','CreditCard_No','PersonalLoan_No']
    rdummies = ['Education_1','SecuritiesAccount_Yes','CDAccount_Yes','Online_Yes','CreditCard_Yes','PersonalLoan_No']
    df_sample5 = df_sample4.copy()
    df_sample5 = df_sample4 .drop(columns=rdummies)

    print (df_sample5.columns.values)
    print (df_sample5)

    # 以上完成Pre-processing
    df4partition = df_sample5.copy()
    testpart_size = 0.2
    df_nontestData, df_testData = train_test_split(df4partition, test_size=testpart_size, random_state=1)
    print(df_nontestData)

    # Logistic Regression with penalty
    DV = 'PersonalLoan_Yes'
    y = df_nontestData[DV]
    X = df_nontestData.drop(columns = [DV])

    alpha = 10
    clf = LogisticRegression(C=1/alpha, penalty='l1',solver='saga', max_iter=200,random_state=1).fit(X,y)
    # 使用l1的penalty,saga类似于penalty type， max_iter是迭代次数, 同样目的是拿出一个合适的alpha

    print (summary_corf(clf))
    # 接下来搞Cross Validation
    kfolds = 5
    min_alpha = 0.001
    max_alpha = 100
    n_candidates = 1000  # 完全没听到

    alpha_list = list(np.linspace(min_alpha,max_alpha,num = n_candidates))
    C_list = list(1/np.linspace(min_alpha,max_alpha,num = n_candidates))

    print (C_list)  # 0.001到100平均分1000个数字

    clf_optimal = LogisticRegressionCV(Cs=C_list, cv=kfolds, penalty='l1',solver='saga',max_iter= 200, random_state=1,
                                       n_jobs=-1).fit(X,y)
    print (summary_corf(clf_optimal))
    print (1/clf_optimal.C_)  # 整出alpha

    y_test_actual = df_testData[DV]
    X_test = df_testData.drop(columns=[DV])

    y_test_predicted = clf_optimal.predict(X_test)
    print(y_test_predicted)
    print (y_test_actual)
    print(metrics.confusion_matrix(y_test_actual,y_test_predicted)) #混淆矩阵，判断准确率
    print(clf_optimal.score(X_test, y_test_actual)) # 输出准确率

    df_newdata = pd.read_csv('/Users/simon/JBWorkspace/BUAN/Session3/Ref_File/PersonalLoan_NEWDATA.csv')

    Original_DV = 'PersonalLoan'
    cpredictor_list = cvar_list.copy()
    cpredictor_list.remove(Original_DV)
    print(cpredictor_list)

    df_newdata_sample1 = df_newdata.drop(columns=rvar_list)

    df_newdata_sample2 = df_newdata_sample1.copy()
    df_newdata_sample2[cpredictor_list] = df_newdata_sample1[cpredictor_list].astype('category')
    df_newdata_sample2[nvar_list] = df_newdata_sample1[nvar_list].astype('float64')

    historical_sample_mean = df_sample1[nvar_list].mean()
    historical_sample_std = df_sample1[nvar_list].std()

    df_newdata_sample3 = df_newdata_sample2.copy()
    df_newdata_sample3[nvar_list] = (df_newdata_sample3[nvar_list]-historical_sample_mean[nvar_list])/historical_sample_std[nvar_list]

    df_newdata_sample4 = pd.get_dummies(df_newdata_sample3,prefix_sep='_')
    print(df_newdata_sample4.columns.values)

    # 添加新列到测试数据里
    df_newdata_sample5 = df_newdata_sample4.copy()
    df_newdata_sample5['Education_2'] = 0
    print(df_newdata_sample5.columns.values)

    predicted_PersonalLoan = clf_optimal.predict(df_newdata_sample5)
    predicted_PersonalLoan_1 = clf_optimal.predict_proba(df_newdata_sample5)[:,1] # 取第二个数据

    print (predicted_PersonalLoan)
    print(predicted_PersonalLoan_1)







