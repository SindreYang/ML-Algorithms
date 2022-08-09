# -*- coding:utf8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import random
#调包实现
from sklearn import linear_model
import numpy as np
filename=r'd:\Algorithms\Regression.txt'
X=[]
y=[]
with open(filename,'r') as f:
    for line in f:
        data=[float(i) for  i  in line.split('\t')]
        xt, yt = data [:-1], data [-1]
        X.append (xt)
        y.append (yt)

# 训练并划分数据
print(len(X))
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# 训练数据
X_train = np.array(X[:num_training])
y_train = np.array(y[:num_training])
X_test = np.array(X[num_training:])

#定义线性回归模型
linear_regressor=linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred = linear_regressor.predict(X_test)


#画图
plt.figure()
plt.plot(X,y,'go', label='Original Data')
plt.plot(X_test,y_test_pred,'r-',linewidth=3,label='Regression Curve')
plt.xlabel('Times')
plt.ylabel('Ratio/Step')
plt.grid(True)
plt.show()

