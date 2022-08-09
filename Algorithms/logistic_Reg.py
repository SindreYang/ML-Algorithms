from numpy import *
import os
import  numpy as np
from  sklearn.datasets import  load_breast_cancer
import matplotlib.pyplot as plt
import random

# 从文件中读入训练样本的数据
def loadDataSet():
    dataMat = [random.normalvariate (0, 1) for _ in range (10)]
    labelMat = [int(i) for i in dataMat]
    return dataMat,labelMat
    
    
def sigmoid(X):
    return 1.0/(1+exp(-X))

# 梯度下降法求回归系数a
def gradAscent(dataMatIn, classLabels):
    dataMatrix = mat (dataMatIn)  # 转换成numpy中的矩阵, X, 90 x 3
    labelMat = mat (classLabels).transpose ()
    n = len(dataMatIn)  # m=90, n=3
    alpha = 0.001  # 学习率
    maxCycles = 1000
    weights = ones((n, 1))  # 初始参数, 3 x 1
    for _ in range(maxCycles):
        h = sigmoid(dataMatrix * weights)     # 模型预测值, 90 x 1
        error = h - labelMat.T              # 真实值与预测值之间的误差, 90 x 1
        temp = dataMatrix.transpose() * error  # 所有参数的偏导数, 3 x 1
        weights = weights - alpha * temp  # 更新权重
    return weights

# 测试函数
def test_logistic_regression():
    dataArr, labelMat = loadDataSet()  # 读入训练样本中的原始数据
    A = gradAscent(dataArr, labelMat)  # 回归系数a的值
    h = sigmoid(mat(dataArr)*A)  # 预测结果h(a)的值
    print()
    plt.plot(dataMat,labelMat)
    plt.show()
    # plotBestFit(A.getA())

test_logistic_regression()