#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:31:32 2017

@author: Andrew
"""

import xlrd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from itertools import combinations
from sklearn import linear_model
#import statsmodels.formula.api as sm
#import pandas

def transform(sample_col):
    row = len(sample_col)
    col = len(sample_col[0])
    return [[sample_col[i][j] for i in range(row)] for j in range(col)]

data=xlrd.open_workbook('SFE_Dataset.xlsx')
table = data.sheets()[0]
nrows = table.nrows
ncols = table.ncols
predictors = []

'''
pre-processing
'''
for i in range(ncols - 1):
    predictors.append(table.col_values(i)[1:])
headers = table.row_values(0)[0:-1]
SFE = table.col_values(ncols - 1)[1:]


k = 0
for j in range(len(predictors)):
    zeroCT = 0
    for value in predictors[j - k]:
        if value == 0:
            zeroCT += 1
        if zeroCT / (nrows - 1) > 0.4:
            del(predictors[j - k])
            del(headers[j - k])
            k += 1
            break
Samples = transform(predictors)
k = 0
for i in range(len(Samples)):
    for element in Samples[i - k]:
        if element == 0:
            del(Samples[i - k])
            del(SFE[i - k])
            k += 1
            break
'''
linear regression Assign 1.1
'''

Coef_list = []
r2_list = []
MSE = []
Predictors = transform(Samples)
feature_X = np.asarray(Predictors)
for i in range(len(Predictors)):
    feature_test = feature_X[i]
    feature_test.shape = (211,1)  
    regr = linear_model.LinearRegression()
    regr.fit(feature_test, SFE)
    pre_SFE = regr.predict(feature_test)
    r2_list.append(regr.score(feature_test, SFE))
    Coef_list.append(regr.coef_)
    MSE.append(mean_squared_error(SFE, pre_SFE))
    plt.figure(i)
    plt.scatter(feature_test, SFE, color = 'blue')
    plt.plot(feature_test, pre_SFE, color = 'black')


'''
exhaustive and Sequential forward search
'''

def regre_score(feature_list, SFE, i):
    feature_array = np.asarray(feature_list)
    feature_array.shape = (211, i)
    regr_exhaus = linear_model.LinearRegression()
    regr_exhaus.fit(feature_array, SFE)
    #pre_2_SFE = regr_exhaus.predict(feature_array)
    return(regr_exhaus.score(feature_array, SFE))    

def regre_Coef(feature_list, SFE, i):
    feature_array = np.asarray(feature_list)
    feature_array.shape = (211, i)
    regr_exhaus = linear_model.LinearRegression()
    regr_exhaus.fit(feature_array, SFE)
    return(regr_exhaus.coef_)

def regre_MSE(feature_list, SFE, i):
    feature_array = np.asarray(feature_list)
    feature_array.shape = (211, i)
    regr_exhaus = linear_model.LinearRegression()
    regr_exhaus.fit(feature_array, SFE)
    pre_2_SFE = regr_exhaus.predict(feature_array)
    return(mean_squared_error(SFE, pre_2_SFE))

'''
exhaustive r2
'''

exhaus_r2_dic = {}
exhaus_Coef_dic = {}
exhaus_MSE_dic = {}
exhaus_adjr2_dic = {}
for i in range(1, 6):
    y = combinations(range(7), i)
    for j in y:
        fea_exhaustive = []
        exhaus_list = []
        for k in j:
            fea_exhaustive.append(Predictors[k])            
            exhaus_list.append(k)
            exhaus_tp = tuple(exhaus_list)
        fea_exhaustive = transform(fea_exhaustive)
        exhaus_r2_dic[exhaus_tp] = regre_score(fea_exhaustive, SFE, i)
        r2 = regre_score(fea_exhaustive, SFE, i)
        exhaus_Coef_dic[exhaus_tp] = regre_Coef(fea_exhaustive, SFE, i)
        exhaus_MSE_dic[exhaus_tp] = regre_MSE(fea_exhaustive, SFE, i)
        exhaus_adjr2_dic[exhaus_tp] = r2 - (len(exhaus_list) - 1) * (1 - r2) / (211 - len(exhaus_list))
        
best_r2_exhaustive = {}
best_MSE_exhaustive = {}
best_adjr2_exhaustive = {}
for i in range(1, 6):
    max_r2_header = None
    min_Coef_header = None
    min_MSE_header = None
    max_r2 = None
    min_Coef = None
    min_MSE = None
    max_adjr2_header = None
    max_adjr2 = None
    for a, b in exhaus_r2_dic.items():
        if len(a) == i:
            if max_r2 == None or b > max_r2:
                max_r2_header = a
                max_r2 = b

    for a, b in exhaus_MSE_dic.items():
        if len(a) == i:
            if min_MSE == None or b < min_MSE:
                min_MSE_header = a
                min_MSE = b
    for a, b in exhaus_adjr2_dic.items():
        if len(a) == i:
            if max_adjr2 == None or b > max_adjr2:
                max_adjr2_header = a
                max_adjr2 = b
    
    best_r2_exhaustive[max_r2_header] = max_r2                
    best_adjr2_exhaustive[max_adjr2_header] = max_adjr2
    best_MSE_exhaustive[min_MSE_header] = min_MSE
    
SFS_X = [2]
SFS_left = [0, 1, 3, 4, 5, 6]
SFS_predictor = []
SFS_predictor.append(Predictors[2])
SFS_r2_max_list = []
SFS_r2_max_list.append(r2_list[2])
for i in range(4):
    SFS_r2_max = None
    ct = 0
    for element in SFS_left:
        ct += 1
        SFS_X.append(element)
        SFS_predictor.append(Predictors[element])
        SFS_predictor_ = transform(SFS_predictor)
        r2_score = regre_score(SFS_predictor_, SFE, i + 2)
        if SFS_r2_max == None or r2_score > SFS_r2_max:
            SFS_r2_max = r2_score
            SFS_X[i + 1] = element
            SFS_predictor[i + 1] = Predictors[element]
            max_ele = element
        SFS_X.remove(element)
        SFS_predictor.remove(Predictors[element])
        if ct == 6-i:
            SFS_left.remove(max_ele)
            SFS_X.append(max_ele)
            SFS_predictor.append(Predictors[max_ele])
            SFS_r2_max_list.append(SFS_r2_max)

           
'''
ridge regression
λ = 50, 30, 15, 7, 3, 1, 0.30, 0.10, 0.03, 0.01.
'''
lamda = [50, 30, 15, 7, 3, 1, 0.30, 0.10, 0.03, 0.01]
sample_array = np.asarray(Samples)
sample_array.shape = (211, 7)
Coef_ridge = []
Coef_lasso = []
for i in lamda:
    ridge = linear_model.Ridge(alpha = i)
    ridge.fit(sample_array, SFE)
    Coef_ridge.append(ridge.coef_)
    lasso = linear_model.Lasso(alpha = i)
    lasso.fit(sample_array, SFE)
    Coef_lasso.append(lasso.coef_)
plt.figure(1)
plt.plot(lamda, Coef_ridge)
plt.title('Ridge Regression')
plt.xlabel('λ')
plt.ylabel('Coefficient')
plt.figure(2)
plt.plot(-np.log(lamda), Coef_lasso)
plt.title('Lasso')
plt.xlabel('λ /dB')
plt.ylabel('Coefficient')

   

            
        
       
        
        
        

    


        
