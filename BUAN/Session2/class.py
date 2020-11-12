import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LassoCV  # cross_validation
import sklearn as sk
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def column_nuique_values(x_column):
    return list(x_column.unique())
    pass


if __name__ == "__main__":
    '''
    Part-1
    We have many candidates for the best prediction model. So, how to choose.
    Based on a mounts of predictors.
    High predictive performance for future NEW DATA. 
    
    Can we "peep" a model's predictive performance for new data before we have new data?
    - Data partition. Training partition and Validation partition.
    
    A critical assumption----Historical data and future new data have the same underlying data generation process.(
    相同的底层数据生成过程) 
    
    Tree partitions.
    - training, validation(Model Tuning), test(Evaluate and Report)
    - Random sample & non-overlapped(随机样本以及无重叠)
    - Always needed for building prediction models.
    
    Part-2
    Liner regression
    -Easy to interpret -Fast to compute -Linearity is a good approximation(近似) 
    produce reasonably good prediction model
    Two criteria foe selecting prediction models
    - High predictive performance - Parsimonious structure（越牛逼越好，用的属性越少越好）
    
    ASE , RASE(最小二乘法的验证)
    
    Part-3
    How to determine the importance of a predictor?
    - Impose a penalty cost. 
    - The penalty cost is the same across all the predictors.
    What is the optimal penalty level a ?
    - The one that results in the model candidate tha has the highest predictive performance over the validation data.(ASE)
     
    Part-4
    LASSO in Python
    LassoCV function in the scikit-learn package - LASSO with cross-validation
    '''
    url = '/Users/simon/JBWorkspace/BUAN/Session1/Ref_File/UsedCar_MissingValue(1)(1).csv'
    df = pd.read_csv(url)
    print(df.head(10))
    print(df.columns.values)
    print(df.shape)
    x_var = 'KM'
    y_var = 'Price'
    df.plot.scatter(x=x_var, y=y_var, legend=False)

    df4heatmaps = df
    # Compute the correlation matrix
    corr = df4heatmaps.corr()
    # Print the correlation matrix
    print(corr)
    # Draw the heatmap
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)

    x_var = 'Fuel_Type'
    y_var = 'Price'
    df4ssb = df

    # Generate the side-by-side box plot. ax is the side-by-side boxplot object
    ax = df4ssb.boxplot(column=y_var, by=x_var)
    # Set the label on the y-axis using set_ylabel method of the object ax
    ax.set_ylabel(y_var)

    # Part 3 Missing Value Imputation

    # Show the number of missing values for each variable in the data frame
    df.isnull().sum()

    # What if we drop all the observations that contains missing values
    print(df.dropna())

    # Drop the observations that contain missing dependent variable values
    # Placeholder variable is DV
    DV = 'Price'
    df_sample1 = df.dropna(subset=[DV])

    # print(df_sample1)

    # Separate all the variables into two lists for future column indexing
    # One for numerical, the other for categorical
    cvar_list = ['Fuel_Type', 'Metallic', 'Automatic', 'Doors']
    nvar_list = ['Price', 'Age', 'KM', 'HP', 'CC', 'Quarterly_Tax', 'Weight']

    # Impute numerical missing values using sample median
    df_sample2 = df_sample1.copy()
    df_sample2[nvar_list] = df_sample1[nvar_list].fillna(value=df_sample1[nvar_list].median())

    # Get the unique values of the categorical variable Fuel_Type
    df_sample2['Fuel_Type'].unique()


    # Get the unique values of each categorical variables in the data frame
    def column_unique_values(x_column):
        return list(x_column.unique())


    df_sample2[cvar_list].apply(column_unique_values)

    # Replace the irregular values with the null values which represent missing values in Python
    # Placeholder variable: irregular_var, irregular_value
    irregular_var = 'Fuel_Type'
    irregular_value = '?'

    # pd.np.nan refers to null values in Python
    df_sample3 = df_sample2.copy()
    df_sample3[irregular_var] = df_sample2[irregular_var].replace(irregular_value, np.nan)

    # Impute the categorical missing values using sample mode
    df_sample4 = df_sample3.copy()
    df_sample4[cvar_list] = df_sample3[cvar_list].fillna(value=df_sample3[cvar_list].mode().loc[0, :])

    # Check if there is any missing value left
    df_sample4.isnull().sum()

    # %%

    # Part 4 Variable transformation

    # Standardize the numerical variables
    df_sample5 = df_sample4.copy()
    df_sample5[nvar_list] = (df_sample4[nvar_list] - df_sample4[nvar_list].mean()) / df_sample4[nvar_list].std()

    # Set the datatype for the variables in the cvar_list to be categorical in Python
    # Set the datatype for the variables in the nvar_list to be numerical in Python
    df_sample6 = df_sample5.copy()
    df_sample6[cvar_list] = df_sample5[cvar_list].astype('category')
    df_sample6[nvar_list] = df_sample5[nvar_list].astype('float64')

    # Convert the categorical variables into dummies (Step 1 of dummy coding)
    # prefix_sep is the sympol used to create the dummy variable names.
    # For example, if we choose underscore _, the dummy variable name will be Fuel_Type_Diesel
    # If we choose dash -, it will be Fuel_Type-Diesel
    df_sample7 = df_sample6.copy()
    df_sample7 = pd.get_dummies(df_sample7, prefix_sep='_')

    # Remove the redundant dummies (Step 2 of dummy coding)
    # Placeholder variable: rdummies
    rdummies = ['Fuel_Type_Petrol', 'Metallic_No', 'Automatic_N', 'Doors_5.0']
    df_sample8 = df_sample7.copy()
    df_sample8 = df_sample8.drop(columns=rdummies)

    # Get the remaining variable list after the variable transformation
    print(df_sample8.columns.values)

    # Display the milestone dataframe. Compare it with the original dataframe.
    print(df_sample8)
    print(df)


    # %%
    # Part 5 Data Partition (以上是前置pre-processing)
    def summary_corf(model_object):  # 利用alpha去进行迭代
        n_predictors = X.shape[1]
        model_coef = pd.DataFrame(model_object.coef_.reshape(1, n_predictors), columns=X.columns.values)
        model_coef['Intercept'] = model_object.intercept_  # 神秘模型，等会要看懂，线性方程的截距
        return model_coef.transpose()
        pass


    df4partation = df_sample8.copy()
    testpart_size = 0.2
    df_nontestData, df_testData = train_test_split(df4partation, test_size=testpart_size,
                                                   random_state=1)  # 记得查一下random_state
    print ("nontestData:")
    print(df_nontestData)
    print ("testData:")
    print(df_testData)

    # Part 6 Lasso analysis

    DV = 'Price'
    y = df_nontestData[DV]
    X = df_nontestData.drop(columns=[DV])

    kfolds = 5  # 数据等分数，分别作为train和验证
    clf_optimal = LassoCV(cv=kfolds, random_state=1, n_jobs=-1).fit(X, y)  # 寻找最合适的alpha， Lasso根据你给的alpha来
    print(clf_optimal)
    print(clf_optimal.alpha_)

    alpha = 0.001  # Choose a penalty level
    alpha_1 = clf_optimal.alpha_  # 来自LassoCV的alpha
    clf = Lasso(alpha=alpha_1, random_state=1).fit(X, y)  # X,predictors
    print(clf)
    model_object = clf

    print(summary_corf(model_object))
    print("xxxxx")
    print(summary_corf(clf_optimal))

    # 计算ASE
    y_test_actual = df_testData[DV]
    X_test = df_testData.drop(columns=[DV])
    y_test_predicted = clf.predict(X_test)
    n_obs_test = df_testData.shape[0]
    ASE_test = (sum((y_test_actual - y_test_predicted) ** 2)) / n_obs_test
    print(ASE_test)  # How much deviation is OK?

    DV = 'Price'
    y = df_nontestData[DV]
    X = df_nontestData.drop(columns=[DV])

    y_nontest_actual = y
    y_nontest_predicted = clf_optimal.predict(X)
    n_obs_nontest = df_nontestData.shape[0]
    ASE_non = (sum((y_nontest_actual - y_nontest_predicted) ** 2)) / n_obs_nontest
    print(ASE_non)  # Why can we do this?

    # %%
    # Part 7 Score the new data
    print("")
    url = '/Users/simon/JBWorkspace/BUAN/Session2/Ref_File/UsedCar_NewData.csv'
    td = pd.read_csv(url)
    print (td)

    print (td)
    df_newdata = td.copy()
    print (df_newdata.isnull().sum()) #检查missing
    print (df_newdata.isnull())

    npredictor_list = nvar_list.copy()
    npredictor_list.remove(DV)
    print(npredictor_list) # 拷贝一下variable列表
    print(cvar_list)

    df_newdata_sample1 = df_newdata.copy()
    df_newdata_sample1[cvar_list] = df_newdata[cvar_list].astype('category')
    df_newdata_sample1[npredictor_list] = df_newdata[npredictor_list].astype('float64') #数据类型转换

    historical_sample_mean = df_sample4[nvar_list].mean()
    historical_sample_std = df_sample4[nvar_list].std()

    df_newdata_sample2 = df_newdata_sample1.copy()
    df_newdata_sample2[npredictor_list] = (df_newdata_sample2[npredictor_list] - historical_sample_mean[npredictor_list])/\
                                          historical_sample_std[npredictor_list] # numerical标准化
    df_newdata_sample3 = pd.get_dummies(df_newdata_sample2,prefix_sep='_') # categorical状态编码
    print(df_newdata_sample3.columns.values)
    print(df_newdata_sample3)

    rdummies_2 = ['Metallic_No','Automatic_N','Doors_5']
    df_newdata_sample4 = df_newdata_sample3.drop(columns=rdummies_2)  #去除多余变化

    print(df_newdata_sample4.columns.values)

    df_newdata_sample5 = df_newdata_sample4.rename(columns = {'Doors_3':'Doors_3.0'})
    print(df_newdata_sample5.columns.values)

    predicted_standardizedPrice = clf_optimal.predict(df_newdata_sample5) # 套用之前的clf_optimal,出标准化结果
    print (predicted_standardizedPrice)
    predicted_Price = predicted_standardizedPrice * historical_sample_std[DV] + historical_sample_mean[DV] # 还原
    print(predicted_Price)

    print (historical_sample_std)
    pass
