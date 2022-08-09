# -*- coding: utf-8 -*-

from sklearn.datasets import load_iris
from numpy import *
#load "iris.data" to workspace
data=load_iris()
traindata = loadtxt(data,delimiter = ',',usecols = (0,1,2,3),dtype = float)
trainlabel = loadtxt(data,delimiter = ',',usecols = (list(range(4,5))),dtype = str)
feaname = ["#0","#1","#2","#3"] # feature names of the 4 attributes (features)
feanamecopy = feaname[:]
args = [0,0,0,0]
#args = mean(traindata,axis = 0)


#计算熵
from math import log
def calentropy(label):
    n = label.size # the number of samples
    #print n
    count = {} #create dictionary "count"
    for curlabel in label:
        if curlabel not in list(count.keys()):
            count[curlabel] = 0
        count[curlabel] += 1
    entropy = 0
    #print count
    for key in count:
        pxi = float(count[key])/n #转换 float
        entropy -= pxi*log(pxi,2)
    return entropy

#testcode:
#x = calentropy(trainlabel)


def gini(label):
    n = label.size # the number of samples
    #print n
    count = {} #create dictionary "count"
    for curlabel in label:
        if curlabel not in list(count.keys()):
            count[curlabel] = 0
        count[curlabel] += 1
    pi_c = array(list(count.values()))*1.0/n
    return sum(1-pi_c**2)
    
#testcode:
#print gini(trainlabel)



#根据标签分割数据集"splitfea_idx"
def splitdata(oridata,splitfea_idx):
    arg = args[splitfea_idx] #求所有维度的平均值
    idx_less = [] #创建包含特性小于pivot的数据的新列表 
    idx_greater = [] #具有大于pivot的特性的条目。
    n = len(oridata)
    for idx in range(n):
        d = oridata[idx]
        if d[splitfea_idx] < arg:
            #将newentry添加到newdata_less集合中。
            idx_less.append(idx)
        else:
            idx_greater.append(idx)
    return idx_less,idx_greater

#testcode:2
#idx_less,idx_greater = splitdata(traindata,2)


#根据索引给出数据和标签
def idx2data(oridata,label,splitidx,fea_idx):
    idxl = splitidx[0] #split_less_indices
    idxg = splitidx[1] #split_greater_indices
    labell = []
    labelg = []
    datal = [append(oridata[i][:fea_idx],oridata[i][fea_idx+1:]) for i in idxl]
    datag = [append(oridata[i][:fea_idx],oridata[i][fea_idx+1:]) for i in idxg]
    labell = label[idxl]
    labelg = label[idxg]
    return datal,datag,labell,labelg
    

#选择要分割的最佳分支
def choosebest_splitnode(oridata,label):
    n_fea = len(oridata[0])
    n = len(label)
    #base_entropy = calentropy(label)
    base_gini = gini(label)
    best_gain = -1
    for fea_i in range(n_fea): #计算每个分裂特征下的熵
        #cur_entropy = 0
        cur_gini = 0
        idxset_less,idxset_greater = splitdata(oridata,fea_i)
        prob_less = float(len(idxset_less))/n
        prob_greater = float(len(idxset_greater))/n

        #entropy(value|X) = \sum{p(xi)*entropy(value|X=xi)}
        #cur_entropy += prob_less*calentropy(label[idxset_less])
        #cur_entropy += prob_greater * calentropy(label[idxset_greater])
        #info_gain = base_entropy - cur_entropy #notice gain is before minus after

        cur_gini += prob_less*gini(label[idxset_less])
        cur_gini += prob_greater * gini(label[idxset_greater])        
        #info_gain = base_entropy - cur_gini #notice gain is before minus after
        info_gain = base_gini - cur_gini #notice gain is before minus after
        if(info_gain>best_gain):
            best_gain = info_gain
            best_idx = fea_i
    return best_idx  

#testcode:
#x = choosebest_splitnode(traindata,trainlabel)



#基于信息增益创建决策树
def buildtree(oridata, label):
    if label.size==0: #if no samples belong to this branch
        return "NULL"
    listlabel = label.tolist()
    #stop when all samples in this subset belongs to one class
    if listlabel.count(label[0])==label.size:
        return label[0]

    #return the majority of samples' label in this subset if no extra features avaliable
    if len(feanamecopy)==0:
        cnt = {}
        for cur_l in label:
            if cur_l not in list(cnt.keys()):
                cnt[cur_l] = 0
            cnt[cur_l] += 1
        maxx = -1 
        for keys in cnt:
            if maxx < cnt[keys]:
                maxx = cnt[keys]
                maxkey = keys
        return maxkey

    bestsplit_fea = choosebest_splitnode(oridata,label) #get the best splitting feature
    print(bestsplit_fea,len(oridata[0]))
    cur_feaname = feanamecopy[bestsplit_fea] # add the feature name to dictionary
    print(cur_feaname)
    nodedict = {cur_feaname:{}} 
    del(feanamecopy[bestsplit_fea]) #delete current feature from feaname
    split_idx = splitdata(oridata,bestsplit_fea) #split_idx: the split index for both less and greater
    data_less,data_greater,label_less,label_greater = idx2data(oridata,label,split_idx,bestsplit_fea)

    #build the tree recursively, the left and right tree are the "<" and ">" branch, respectively
    nodedict[cur_feaname]["<"] = buildtree(data_less,label_less)
    nodedict[cur_feaname][">"] = buildtree(data_greater,label_greater)
    return nodedict
    
#testcode:
mytree = buildtree(traindata,trainlabel)
print(mytree)

#一个新的样本进行分类
def classify(mytree,testdata):
    if type(mytree).__name__ != 'dict':
        return mytree
    fea_name = list(mytree.keys())[0] #get the name of first feature
    fea_idx = feaname.index(fea_name) #the index of feature 'fea_name'
    val = testdata[fea_idx]
    nextbranch = mytree[fea_name]

    #judge the current value > or < the pivot (average)
    nextbranch = nextbranch[">"] if val>args[fea_idx] else nextbranch["<"]
    return classify(nextbranch,testdata)

#testcode
tt = traindata[0]
x = classify(mytree,tt)
print(x)

    
