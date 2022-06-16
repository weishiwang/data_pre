# -*- coding: utf-8 -*-
"""
Created on Fri May 20 14:00:37 2022

@author: Administrator
"""

import pandas as pd
import numpy as np
import math
import random
import seaborn as sns

# def detect_outliers(data, threshold=3):
#     """离群点检测"""
# #    np.mean()是用来计算均值，np.std()是用来计算标准差
# #    把三倍于数据集的标准差的点设想为噪声数据排除。
#     mean_d = np.mean(data)
#     std_d = np.std(data)
#     outliers = []

#     for y in data:
#         z_score = (y - mean_d) / std_d
#         if np.abs(z_score) > threshold:
#             outliers.append(y)
#     return outliers


# data = pd.read_excel('data2Clean.xlsx')
# col = data.columns.to_list()
# null = data.isnull()
# hangshu=len(data.SD)
# sumnull = data.isnull().sum()
# for i in range(1,len(col)):
#     outliers=detect_outliers(data[col[i]],threshold=3)
#     if(len(outliers)>0):
#         print(col[i])
#         print(len(outliers))
#         print(outliers)
        
#         for j in range(len(outliers)):
#             index=data[(data[col[i]]==outliers[j])].index.tolist()
#             print(index)
#             print(len(index))
#             for k in range(len(index)):
#                 data = data.drop(index[k])
                
# print(len(data.SD))
# print(len(data.SD)/hangshu)

# tmp_file_path='表一除噪后的数据.xlsx'
# data.to_excel(tmp_file_path,sheet_name=tmp_file_path.split('.')[0],index= False)

# print('查看是否还有离群点：')
# for i in range(1,len(col)):
#     outliers=detect_outliers(data[col[i]],threshold=3)
#     if(len(outliers)>0):
#         print(col[i])
#         print(len(outliers))
#         print(outliers)
#     else:
#         print(col[i],'没有离群点')

#=====================================方法二====================================
'''
参考博客：https://blog.csdn.net/weixin_42199542/article/details/106898892
'''
data = pd.read_excel('data2Clean.xlsx')
col = data.columns.to_list()
hangshu=len(data.SD)
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
# print(Q1[0])
jiance = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
outlists = 0
for i in range(len(col)):
    index=data[(jiance[col[i]]==True)].index.tolist()
    if(len(index)>0):
        # print(col[i])
        # print(index)
        outlists=outlists+len(index)
liqunzhanbi = round(outlists/hangshu,2)
if(liqunzhanbi>0.05):
    for i in range(len(col)):
        index=data[(jiance[col[i]]==True)].index.tolist()
        if(len(index)>0):
            # print(col[i])
            # print(i)
            # print(len(index))
            # print(index[0],'第一个')
            # print(index)
            # print(Q1[i-1])
            '''
            (Q1[i-1]因为第一列不用统计
            '''
            for j in range(len(index)):
                print(col[i])
                print(index[j])
                print('被改之前数据',data[col[i]][index[j]])
                if(data[col[i]][index[j]]<Q1[i-1]):
                    data[col[i]][index[j]]=  Q1[i-1]
                else:
                    data[col[i]][index[j]]=  Q3[i-1]
                print('被改后数据',data[col[i]][index[j]])
                
                
tmp_file_path='表1离群点处理后的数据.xlsx'
data.to_excel(tmp_file_path,sheet_name=tmp_file_path.split('.')[0],index= False)
# print(IQR)
# print(IQR[0])
