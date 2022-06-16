# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 09:03:51 2022

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from sklearn import svm
from sklearn import model_selection
from sklearn import metrics
import math



data = pd.read_excel('data2denoise.xlsx')
data2 = pd.read_excel('data2denoise.xlsx')
data['ischange'] = data.apply(lambda x: 0, axis=1)
#d5=data.iloc[50:100]
#自定义函数 e指数形式
predictors = data.columns[1:-1]
x=data[predictors]
y=data.y
#print(data.shape[0])
hang=data.shape[0]
chuli = []
#for i in range(4):
#    cut_place1=int(i*0.25*hang)
#    cut_place2=int((i+1)*0.25*hang)
#    y_test=y.iloc[cut_place1:cut_place2]
#    x_test=x.iloc[cut_place1:cut_place2]
#    if(cut_place1!=0 and cut_place2!=hang):
#        y1=y.iloc[0:cut_place1]
#        y2=y.iloc[cut_place2:hang]
#        
#        x1=x.iloc[0:cut_place1]
#        x2=x.iloc[cut_place2:hang]
#        
#        y_dfs = [y1,y2]
#        y_train = pd.concat(y_dfs)
#        
#        x_dfs = [x1,x2]
#        x_train = pd.concat(x_dfs)
#    elif(cut_place1==0):
#        y_train=y.iloc[cut_place2:hang]
#        x_train=x.iloc[cut_place2:hang]
#    elif(cut_place2==hang):
#        y_train=y.iloc[0:cut_place1]
#        x_train=x.iloc[0:cut_place1]





for i in range(10):

    suijibili = [0.3,0.4,0.5]
    n=i%len(suijibili)
    bili=suijibili[n]
    randomna = random.randint(1,1000)
# 将数据拆分为训练集和测试集
    x_train,x_test,y_train,y_test = model_selection.train_test_split(x, y, test_size = bili, random_state = randomna)
    kernel=['rbf','linear','poly','sigmoid']
    C=[0.1,0.5,1,2,5]
    parameters = {'kernel':kernel,'C':C}
    grid_svc = model_selection.GridSearchCV(estimator = svm.SVC(),
                                            param_grid =parameters,scoring='neg_mean_squared_error',
                                            cv=2,verbose =1)
    # 模型在训练数据集上的拟合faf
    grid_svc.fit(x_train,y_train)
    # 返回交叉验证后的最佳参数值
    #grid_svc.best_params_, grid_svc.best_score_
    
    # 模型在测试集上的预测
    pred_svc = grid_svc.predict(x_test)
#    zhunque=metrics.accuracy_score(y_test,pred_svc)
    zhunque=metrics.mean_squared_error(y_test,pred_svc)
    print('MSE值:',metrics.mean_squared_error(y_test,pred_svc))
    duibi = pd.DataFrame({'实际':y_test,'预测':pred_svc}, columns=['实际','预测'])
    print(duibi)
        
    sumcha = 0
    
    for i in range(duibi.shape[0]):
        a = abs(duibi['实际'][duibi.index[i]]-duibi['预测'][duibi.index[i]])
        sumcha=sumcha + a
           
    sumcha=math.ceil((sumcha/duibi.shape[0]))
    print('误差范围：',sumcha)
    
    sumheli = 0
    for i in range(duibi.shape[0]):
        if(abs(duibi['实际'][duibi.index[i]]-duibi['预测'][duibi.index[i]])<150):
            sumheli=sumheli+1
#    print(metrics.scorer.accuracy_score(y_test, predict))
#    print(metrics.mean_squared_error(y_test, predict))
    print('合格总数：',sumheli)
    hegezhanbi = sumheli / duibi.shape[0]
    print('合格占比：',hegezhanbi)
    # print(duibi.index)
    if(hegezhanbi>0.8):
        for i in range(duibi.shape[0]):
            if(abs(duibi['实际'][duibi.index[i]]-duibi['预测'][duibi.index[i]])>150):
                print('在某行有噪音:',duibi.index[i])
                data.y[duibi.index[i]]=duibi['预测'][duibi.index[i]]
                data.ischange[duibi.index[i]]=1
                chuli.append(duibi.index[i])
    else:
        print('合格不足80%')

qianhouduibi = pd.DataFrame(columns=['原数据','处理后数据'])
qianhouduibi['原数据']=data2['y']
qianhouduibi['处理后数据']=data['y']
qianhouduibi['ischange']=data['ischange']
print('被处理的行',chuli)
    
    
#X_train,X_test,y_train,y_test = model_selection.train_test_split(data[predictors], data.y, 
#                                                                 test_size = 0.25, random_state = 1234)
																 
# 使用网格搜索法，选择线性可分SVM“类”中的最佳C值
#C=[0.05,0.1,0.5,1,2,5]
#parameters = {'C':C}
#grid_linear_svc = model_selection.GridSearchCV(estimator = svm.LinearSVC(),
#                                               param_grid =parameters,
#                                               scoring='accuracy',cv=2,verbose =1)
## 模型在训练数据集上的拟合
#grid_linear_svc.fit(X_train,y_train.astype('int'))
## 返回交叉验证后的最佳参数值
#grid_linear_svc.best_params_, grid_linear_svc.best_score_	
#
## 模型在测试集上的预测
#pred_linear_svc = grid_linear_svc.predict(X_test)





#kernel=['rbf','linear','poly','sigmoid']
#C=[0.1,0.5,1,2,5]
#parameters = {'kernel':kernel,'C':C}
#grid_svc = model_selection.GridSearchCV(estimator = svm.SVC(),
#                                        param_grid =parameters,scoring='neg_mean_squared_error',
#                                        cv=2,verbose =1)
## 模型在训练数据集上的拟合
#grid_svc.fit(X_train,y_train.astype('int'))
## 返回交叉验证后的最佳参数值
##grid_svc.best_params_, grid_svc.best_score_
#
## 模型在测试集上的预测
#pred_svc = grid_svc.predict(X_test)
#zhunque=metrics.accuracy_score(y_test,pred_svc)