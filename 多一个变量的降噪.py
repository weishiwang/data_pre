# -*- coding: utf-8 -*-
"""
Created on Tue May 31 21:15:32 2022

@author: PC
"""

import pandas as pd
import numpy as np
from sklearn import neighbors
import matplotlib.pyplot as plt
import math
import random

data = pd.read_excel('data2denoise.xlsx')
data2 = pd.read_excel('data2denoise.xlsx')
'''
备份data
'''

#X=data.drop(['ID', 'y'], axis=1)
#Y=data.drop(['ID', 'x1', 'x2'], axis=1)
data['ischange'] = data.apply(lambda x: 0, axis=1)
'''
记录被处理的行
'''
chuli = []
for k in range(20):
    from sklearn import model_selection
    # 将数据集拆分为训练集和测试集
    suijibili = [0.3,0.4,0.5]
    n=k%len(suijibili)
    bili=suijibili[n]
    predictors = data.columns[1:-2]
    randomna = random.randint(1,1000)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data[predictors],data.y,
                                                                        test_size = bili, random_state = randomna)
    K = np.arange(1,np.ceil(np.log2(data.shape[0]))).astype(int)
    
    accuracy = []
    for k in K:
        # 使用10重交叉验证的方法，比对每一个k值下KNN模型的预测准确率
        cv_result = model_selection.cross_val_score(neighbors.KNeighborsClassifier(n_neighbors = k, weights = 'distance'), 
                                                    X_train , y_train , cv = 2, scoring='neg_mean_squared_error')
        accuracy.append(cv_result.mean())
    
    # 从k个平均准确率中挑选出最大值所对应的下标    
    arg_max = np.array(accuracy).argmax()
    
    
    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 绘制不同K值与平均预测准确率之间的折线图
    plt.plot(K, accuracy)
    # 添加点图
    plt.scatter(K, accuracy)
    # 添加文字说明
    plt.text(K[arg_max], accuracy[arg_max], '最佳k值为%s' %int(K[arg_max]))
    # 显示图形
    plt.show()
    
    knn_class = neighbors.KNeighborsClassifier(n_neighbors = int(K[arg_max]), weights = 'distance')
    # 模型拟合
    knn_class.fit(X_train, y_train)
    # 模型在测试数据集上的预测
    predict = knn_class.predict(X_test)
    # 构建混淆矩阵
    cm = pd.crosstab(predict,y_test)
    cm
    
    
    from sklearn import metrics
    
    # 模型整体的预测准确率
    # metrics.scorer.accuracy_score(y_test, predict)
    
    # 分类模型的评估报告
    # print(metrics.classification_report(y_test, predict))
    
    
    duibi = pd.DataFrame({'实际':y_test,'预测':predict}, columns=['实际','预测'])
    print(duibi)
    # print(duibi['实际'][duibi.index[0]])
    # print(duibi.shape[0])
    sumcha = 0
    
    for i in range(duibi.shape[0]):
        a = abs(duibi['实际'][duibi.index[i]]-duibi['预测'][duibi.index[i]])
        sumcha=sumcha + a
           
    sumcha=math.ceil((sumcha/duibi.shape[0]))
    print('误差范围：',sumcha)
    
    sumheli = 0
    for i in range(duibi.shape[0]):
        if(abs(duibi['实际'][duibi.index[i]]-duibi['预测'][duibi.index[i]])<200):
            sumheli=sumheli+1
#    print(metrics.scorer.accuracy_score(y_test, predict))
#    print(metrics.mean_squared_error(y_test, predict))
    print('合格总数：',sumheli)
    hegezhanbi = sumheli / duibi.shape[0]
    print('合格占比：',hegezhanbi)
    # print(duibi.index)
    if(hegezhanbi>0.7):
        for i in range(duibi.shape[0]):
            if(abs(duibi['实际'][duibi.index[i]]-duibi['预测'][duibi.index[i]])>200):
                print('在某行有噪音:',duibi.index[i])
                data.y[duibi.index[i]]=duibi['预测'][duibi.index[i]]
                data.ischange[duibi.index[i]]=1
                chuli.append(duibi.index[i])

qianhouduibi = pd.DataFrame(columns=['原数据','处理后数据'])
qianhouduibi['原数据']=data2['y']
qianhouduibi['处理后数据']=data['y']
qianhouduibi['ischange']=data['ischange']
print('被处理的行',chuli)
tmp_file_path='多一个变量降噪处理后的数据.xlsx'
data.to_excel(tmp_file_path,sheet_name=tmp_file_path.split('.')[0],index= False)