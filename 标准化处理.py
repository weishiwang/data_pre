# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 12:03:34 2022

@author: PC
"""


# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler
# import numpy
 
# data = pd.read_excel('data2Clean.xlsx') 
# # print(data)
# data=data.drop(['Targets in test set'], axis=1)
# data = data.dropna() #去除含有空值的行

# r2=StandardScaler().fit_transform(data)#标准化处理
# print(len(r2[0]))
# print(data)
# numpy.savetxt('标准化后数据.csv', r2,delimiter = ',')


# =================================方法二=====================================

from sklearn import preprocessing
import pandas as pd

data = pd.read_excel('data2Clean.xlsx') 
data=data.drop(['Targets in test set'], axis=1)
data = data.dropna() #去除含有空值的行
min_max_normalizer=preprocessing.MinMaxScaler(feature_range=(0,1))
#feature_range设置最大最小变换值，默认（0,1）
scaled_data=min_max_normalizer.fit_transform(data)
#将数据缩放(映射)到设置固定区间
biaozhunhua=pd.DataFrame(scaled_data)
#将变换后的数据转换为dataframe对象
print(biaozhunhua)


'''
参考博客：https://blog.csdn.net/weixin_46031067/article/details/118767432
'''

# ===============================方法三========================================

from sklearn import preprocessing
import pandas as pd

data = pd.read_excel('data2Clean.xlsx') 
data=data.drop(['Targets in test set'], axis=1)
data = data.dropna() #去除含有空值的行
normalizer=preprocessing.scale(data)
#沿着某个轴标准化数据集，以均值为中心，以分量为单位方差
price_frame_normalized=pd.DataFrame(normalizer,columns=['price'])
#将标准化的数据转换为dataframe对象，将列名改为price
print(price_frame_normalized)