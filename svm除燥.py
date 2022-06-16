from sklearn import svm
import pandas as pd
from sklearn import model_selection
from sklearn import metrics

data = pd.read_excel('data2denoise.xlsx')
data2 = pd.read_excel('data2denoise.xlsx')
data['ischange'] = data.apply(lambda x: 0, axis=1)
chuli = []
from sklearn import preprocessing
import numpy as np
from sklearn import neighbors
# 对变量MEDV作对数变换
y = data.y
# 将X变量作标准化处理
predictors = data.columns[1:-1]
X = data[predictors]

# 将数据拆分为训练集和测试集
X_train,X_test,y_train,y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = 1234)

# 构建默认参数的SVM回归模型
#svr = svm.SVR()
#svr = svm.LinearSVR()
#svr = svm.SVC()
#svr = svm.LinearSVC()
# 模型在训练数据集上的拟合
#svr.fit(X_train,y_train)
## 模型在测试上的预测
#pred_svr = svr.predict(X_test)
## 计算模型的MSE
#mmse1=metrics.mean_squared_error(y_test,pred_svr)
#print()
#print()
#print('模型的MSE',metrics.mean_squared_error(y_test,pred_svr))
#
#duibi1 = pd.DataFrame({'实际':y_test,'预测':pred_svr}, columns=['实际','预测'])
#print(duibi1)

epsilon = np.arange(0.1,1.5,0.2)
C= np.arange(100,1000,200)
gamma = np.arange(0.001,0.01,0.002)
parameters = {'epsilon':epsilon,'C':C,'gamma':gamma}
grid_svr = model_selection.GridSearchCV(estimator = svm.SVR(),param_grid =parameters,
                                        scoring='neg_mean_squared_error',cv=2,verbose =1, n_jobs=2)
# 模型在训练数据集上的拟合
grid_svr.fit(X_train,y_train)
# 返回交叉验证后的最佳参数值
print(grid_svr.best_params_, grid_svr.best_score_)


# 模型在测试集上的预测
pred_grid_svr = grid_svr.predict(X_test)
mmse2 = metrics.mean_squared_error(y_test,pred_grid_svr)
# 计算模型在测试集上的MSE值
print()
print()
print('MSE值:',metrics.mean_squared_error(y_test,pred_grid_svr))

duibi = pd.DataFrame({'实际':y_test,'预测':pred_grid_svr}, columns=['实际','预测'])
print(duibi)

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
if(hegezhanbi>0.8):
    for i in range(duibi.shape[0]):
        if(abs(duibi['实际'][duibi.index[i]]-duibi['预测'][duibi.index[i]])>200):
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