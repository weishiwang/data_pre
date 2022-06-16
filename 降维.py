# coding:utf8

from numpy import *
#from loadData import *
import pandas as pd

'''
Description:PCA算法实现
Author：伏草惟存
Prompt: code in Python3 env
'''

'''加载数据集'''    
def chuli(x,y):
    return x[y]

def loadDataSet():
    logp=pd.read_csv(r'logp数据.csv')
#对摩根指纹进行处理的函数

    # 只留下0、1
    logp.摩根指纹 = logp.摩根指纹.astype('str')
    logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace('.',''))
    logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace('[',''))
    logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace(']',''))
    logp.摩根指纹 = logp.摩根指纹.apply(lambda x:x.replace(' ',''))
    
    #单独生成一列数据
    for i in range(len(logp.摩根指纹[0])):
        logp[i]=logp.apply(lambda x: chuli(x['摩根指纹'],i),axis=1)
    
    for i in range(len(logp.摩根指纹[0])):
        logp[i] = logp[i].str[:].astype('float')
    #print(logp.dtypes)
    data=logp
    data=data.drop('ID',axis=1)
    data=data.drop('摩根指纹',axis=1)
    data=data.drop('LogP值',axis=1)
    dataMat=(data.as_matrix())
    return dataMat


'''pca算法
    方差：（一维）度量两个随机变量关系的统计量,数据离散程度，方差越小越稳定
    协方差： （二维）度量各个维度偏离其均值的程度
    协方差矩阵：（多维）度量各个维度偏离其均值的程度

    当 cov(X, Y)>0时，表明X与Y正相关(X越大，Y也越大；X越小Y，也越小。)
    当 cov(X, Y)<0时，表明X与Y负相关；
    当 cov(X, Y)=0时，表明X与Y不相关。

    cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)]/(n-1)
    Args:
        dataMat   原数据集矩阵
        topNfeat  应用的N个特征
    Returns:
        lowDDataMat  降维后数据集
        reconMat     新的数据集空间

1 去除平均值
2 计算协方差矩阵
3 计算协方差矩阵的特征值和特征向量
4 将特征值排序
5 保留前N个最大的特征值对应的特征向量
6 将数据转换到上面得到的N个特征向量构建的新空间中（实现了特征压缩）
'''
def pca(dataMat, topNfeat=9999999):
    # 1 计算每一列的均值
    meanVals = mean(dataMat, axis=0) # axis=0表示列，axis=1表示行
    print('各列的均值：\n', meanVals)

    # 2 去平均值，每个向量同时都减去均值
    meanRemoved = dataMat - meanVals
    print('每个向量同时都减去均值:\n', meanRemoved)

    # 3 计算协方差矩阵的特征值与特征向量，eigVals为特征值， eigVects为
    # rowvar=0，传入的数据一行代表一个样本，若非0，传入的数据一列代表一个样本
    covMat = cov(meanRemoved, rowvar=0)
    eigVals, eigVects = linalg.eig(mat(covMat))
    print('特征值:\n', eigVals,'\n特征向量:\n', eigVects)

    # 4 将特征值排序, 特征值的逆序就可以得到topNfeat个最大的特征向量
    eigValInd = argsort(eigVals) # 特征值从小到大的排序，返回从小到大的index序号
    # print('eigValInd1=', eigValInd)

    # 5 保留前N个特征。-1表示倒序，返回topN的特征值[-1到-(topNfeat+1)不包括-(topNfeat+1)]
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # 重组 eigVects 最大到最小
    redEigVects = eigVects[:, eigValInd]
    print('重组n特征向量最大到最小:\n', redEigVects.T)

    # 6 将数据转换到新空间
    # print( "---", shape(meanRemoved), shape(redEigVects))
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print('lowDDataMat=', lowDDataMat)
    # print('reconMat=', reconMat)
    return lowDDataMat, reconMat


if __name__ == "__main__":
    # 1 加载数据，并转化数据类型为float
    logp=pd.read_csv(r'logp数据.csv')
    dataMat = loadDataSet()
    print('加载原始特征数据:\n',dataMat)

    # 2 主成分分析降维特征向量设置
    lowDmat, reconMat = pca(dataMat,5)
    print('PCA降维前的数据规模如下:\n',shape(dataMat))
    print('PCA降维后的数据规模如下:\n',shape(lowDmat))
    newdata=pd.DataFrame(lowDmat)
    logp.LogP值 = logp.LogP值.apply(lambda x:x.replace('[',''))
    logp.LogP值 = logp.LogP值.apply(lambda x:x.replace(']',''))
    logp.LogP值 = logp.LogP值.astype('float')
    newdata['LogP值']=logp['LogP值']
    
    # 只需要2个特征向量，和原始数据一致，没任何变化
    # lowDmat, reconMat = pca(dataMat, 2)
    # print(shape(lowDmat))



