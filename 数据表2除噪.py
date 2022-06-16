# -*- coding: utf-8 -*-
"""
Created on Tue May 31 20:02:01 2022

@author: PC
"""


import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.read_excel('data2Clean.xlsx',sheet_name = 1)

    
#定义x、y散点坐标
x = data.X1
x = np.array(x)
# print('x is :\n',x)
y = data.Y
y = np.array(y)
# print('y is :\n',y)
#用3次多项式拟合
f1 = np.polyfit(x, y, 3)
print('f1 is :\n',f1)
    
p1 = np.poly1d(f1)
print('p1 is :\n',p1)
    
#也可使用yvals=np.polyval(f1, x)
yvals = p1(x) #拟合y值
print('yvals is :\n',yvals)
#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('polyfitting')
plt.show()
num=0
for i in range(len(y)):
    z=abs(y[i]-yvals[i])
    num=num+z
num=math.ceil(num/len(y))
print('离群范围:',num)
for i in range(len(y)):
    z=abs(y[i]-yvals[i])
    if(z>num):
        print('第',i+1,'行有噪音')
        data.Y[i]=int(yvals[i])


tmp_file_path='降噪后的数据.xlsx'
data.to_excel(tmp_file_path,sheet_name=tmp_file_path.split('.')[0],index= False)


#==========================方法二==============================================
#自定义函数
# def func(x, a, b):
#      return a*x+b
    
# #定义x、y散点坐标
# x = data.X1
# x = np.array(x)
# print('x is :\n',x)
# y = data.Y
# y = np.array(y)
# print('y is :\n',y)
    
# #非线性最小二乘法拟合
# popt, pcov = curve_fit(func, x, y)
# #获取popt里面是拟合系数
# print(popt)
# a = popt[0]
# b = popt[1]

# yvals = func(x,a,b) #拟合y值
# print('popt:', popt)
# print('系数a:', a)
# print('系数b:', b)

# print('系数pcov:', pcov)
# print('系数yvals:', yvals)
# #绘图
# plot1 = plt.plot(x, y, 's',label='original values')
# plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4) #指定legend的位置右下角
# plt.title('curve_fit')
# plt.show()

# num=0
# for i in range(len(y)):
#     z=abs(y[i]-yvals[i])
#     num=num+z
# num=math.ceil(num/len(y))
# print('离群范围:',num)
# for i in range(len(y)):
#     z=abs(y[i]-yvals[i])
#     if(z>num):
#         print('第',i+1,'行有噪音')
#         data.Y[i]=int(yvals[i])


# tmp_file_path='降噪后的数据.xlsx'
# data.to_excel(tmp_file_path,sheet_name=tmp_file_path.split('.')[0],index= False)