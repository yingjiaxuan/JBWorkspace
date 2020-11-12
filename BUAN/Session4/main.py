import pandas as pd
import numpy as np

if __name__ == "__main__":
    '''
    Issues with accuracy measure
    - It assumes equal importance between predicting 1 as 1 and predicting 0 as 0.
    - Accuracy favors the naive model which always predicts 0.
    However
    - What if decision weights are unknown?
    - What if no decision is involved in the first place?
    
    Instead of accutacy, we use True Positive Rate(TPR) as the criterion.
    - TPR: For all the actual "1"s, how many percent is correctly predicted as '1'? --- TPR = a/(a+b)
    
    In addition to TPR
    - We also add False Positive Rate (FPR) to be part of the criterion.  FPR = c/(c+d)
    We need high TPR and low FPR.
    
    However, it is difficult to operate on two performance metrics. Can we consolidate(合并) them into one overall measure?
    
    A very important concept. ROC Curve.
    相同的逻辑回归模型，但是采用不同的cut-off，所以会产生一条 FPR-TPR曲线
    Area Under Curve ———— AUC （indicates a models' performance) —— 说人话，面积越大，模型效果越好，至于具体选哪一个TPR，FPR的妥协点，就看profit具体咋算
    
    Set scoring = 'roc_auc' for LogisticRegressionCV #################
    '''
    pass