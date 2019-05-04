
import copy
import numpy as np
import pandas as pd
import random as rd
import scipy.spatial.distance as dist
from sklearn.model_selection import train_test_split
from sklearn import svm,datasets
import matplotlib.pyplot as plt

#载入数据
data1=datasets.load_iris()
X=data1.data[:,:2]
y=data1.target
#分割数据集
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33)
#构建模型
models = (svm.SVC(kernel='linear', C=1.0),
          svm.SVC(kernel='rbf', gamma=0.7, C=1.0),
          svm.SVC(kernel='poly', degree=3, C=1.0))
models = (clf.fit(X, y) for clf in models)

#画图准备
titles = ('SVC线性核' ,
          'SVCRBF核' ,
          'SVC多项式（指数3）内核')
fig, sub = plt.subplots(3, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]

x_min, x_max = X0.min() - 1, X0.max() + 1
y_min, y_max = X1.min() - 1, X1.max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),np.arange(y_min, y_max, .02))

def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('长度')
    ax.set_ylabel('宽度')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()

#######################################################################
#######################################################################
