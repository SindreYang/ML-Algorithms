# -*- coding:utf8 -*-

import math
import matplotlib.pyplot as plt
import numpy as np
import random
# def f(w, x):
    # N = len(w)
    # i = 0
    # y = 0
    # while i < N - 1:
        # y += w[i] * x[i]
        # i += 1
    # y += w[N - 1]  # 常数项
    # return y


# def gradient(data, w, j):
    # M = len(data)   # 样本数
    # N = len(data[0])
    # i = 0
    # g = 0   # 当前维度的梯度
    # while i < M:
        # y = f(w, data[i])
        # if (j != N - 1):
            # g += (data[i][N - 1] - y) * data[i][j]
        # else:
            # g += data[i][N - 1] - y
        # i += 1
    # return g / M


# def isSame(a, b):
    # n = len(a)
    # i = 0
    # while i < n:
        # if abs(a[i] - b[i]) > 0.00001:
            # return False
        # i += 1
    # return True


# def fw(w, data):
    # M = len(data)   # 样本数
    # N = len(data[0])
    # i = 0
    # s = 0
    # while i < M:
        # y = data[i][N - 1] - f(w, data[i])
        # s += y ** 2
        # i += 1
    # return s / 2


# def numberProduct(n, vec, w):
    # N = len(vec)
    # i = 0
    # while i < N:
        # w[i] += vec[i] * n
        # i += 1


# def assign(a):
    # L = []
    # for x in a:
        # L.append(x)
    # return L


# # a = b
# def assign2(a, b):
    # i = 0
    # while i < len(a):
        # a[i] = b[i]
        # i += 1


# def dotProduct(a, b):
    # N = len(a)
    # i = 0
    # dp = 0
    # while i < N:
        # dp += a[i] * b[i]
        # i += 1
    # return dp


# # w当前值；g当前梯度方向；a当前学习率；data数据
# def calcAlpha(w, g, a, data):
    # c1 = 0.3
    # now = fw(w, data)
    # wNext = assign(w)
    # numberProduct(a, g, wNext)
    # next = fw(wNext, data)
    # # 寻找足够大的a，使得h(a)>0
    # count = 30
    # while next < now:
        # a *= 2
        # wNext = assign(w)
        # numberProduct(a, g, wNext)
        # next = fw(wNext, data)
        # count -= 1
        # if count == 0:
            # break

    # # 寻找合适的学习率a
    # count = 50
    # while next > now - c1*a*dotProduct(g, g):
        # a /= 2
        # wNext = assign(w)
        # numberProduct(a, g, wNext)
        # next = fw(wNext, data)

        # count -= 1
        # if count == 0:
            # break
    # return a


# def normalize(g):
    # s = 0
    # for x in g:
        # s += x * x
    # s = math.sqrt(s)
    # i = 0
    # N = len(g)
    # while i < N:
        # g[i] /= s
        # i += 1


# def calcCoefficient(data, listA, listW, listLostFunction):
    # N = len(data[0])  # 维度
    # w = [0 for i in range(N)]
    # wNew = [0 for i in range(N)]
    # g = [0 for i in range(N)]

    # times = 0
    # alpha = 100.0  # 学习率随意初始化
    # while times < 10000:
        # j = 0
        # while j < N:
            # g[j] = gradient(data, w, j)
            # j += 1
        # normalize(g)  # 正则化梯度
        # alpha = calcAlpha(w, g, alpha, data)
        # #alpha=0.02
        # numberProduct(alpha, g, wNew)

        # print(("times,alpha,fw,w,g:\t", times, alpha, fw(w, data), w, g))
        # if isSame(w, wNew):
            # break
        # assign2(w, wNew)  # 更新权值
        # times += 1

        # listA.append(alpha)
        # listW.append(assign(w))
        # listLostFunction.append(fw(w, data))
    # return w


# def sigmoid(X):
    # return 1.0/(1+np.exp(-X))

# if __name__ == "__main__":
    # fileData = open(r'C:\Users\sindre\Desktop\Algorithms\Regression.txt')
    # data = []
    # for line in fileData:
        # d = list(map(float, line.split('\t')))
        # data.append(d)
    # fileData.close()

    # listA = []  # 每一步的学习率
    # listW = []  # 每一步的权值
    # listLostFunction = []  # 每一步的损失函数值
    # w = calcCoefficient(data, listA, listW, listLostFunction)

    # # 绘制学习率
    # plt.figure()
    # plt.plot(listA, 'r-', linewidth=2)
    # plt.plot(listA, 'go')
    # plt.xlabel('Times')
    # plt.ylabel('Ratio/Step')
    # plt.grid(True)
    # plt.show()

    # # 绘制损失
    # listLostFunction.pop(0)
    # plt.figure()
    # plt.plot(listLostFunction, 'r-', linewidth=2)
    # plt.plot(listLostFunction, 'go')
    # plt.xlabel('Times')
    # plt.ylabel('Loss Value')
    # plt.grid(True)
    # plt.show()

    # # 绘制权值

    # X = []
    # Y = []
    # for d in data:
        # X.append(d[0])
        # Y.append(d[1])
    # plt.figure()
    # plt.plot(X, Y, 'go', label='Original Data', alpha=0.75)
    # plt.grid(True)
    # x = [min(X), max(X)]
    # y = [w[0] * x[0] + w[1], w[0] * x[1] + w[1]]
    # plt.plot(x, y, 'r-', linewidth=3, label='Regression Curve')
    # plt.legend(loc='upper left')
    # plt.show()


    
    # dataMat = []
    # for i in range (10):
        # dataMat.append (random.normalvariate (0, 1))
    # labelMat = [int (i) for i in dataMat]
    # dataMat=np.array(dataMat)
    # print((type(dataMat)))
    # plt.figure ()
    # plt.plot (dataMat, labelMat, 'go', label='Original Data', alpha=0.75)
    # plt.grid (True)
    # x = [min (dataMat), max (dataMat)]
    # y = [sigmoid(dataMat*w[0],dataMat[1]*w[1])]
    # plt.plot (x, y, 'r-', linewidth=3, label='LR')
    # plt.legend (loc='upper left')
    # plt.show ()
    
#调包实现
from sklearn import linear_model
import numpy as np
filename=r'd:\Algorithms\Regression.txt'
X=[]
y=[]
with open(filename,'r') as f :
    for line in f.readlines():
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

