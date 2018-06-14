from numpy import *
from math import log
'''
@author gsj   18/
/***树回归****/
需要注意的是，Python语言不用考虑内存分配问题。Python语言在函数中传递的是列表的
引用，在函数内部对列表对象的修改，将会影响该列表对象的整个生存周期。为了消除这个不良
影响，我们需要在函数的开始声明一个新列表对象。因为该函数代码在同一数据集上被调用多次，
为了不修改原始数据集，创建一个新的列表对象。数据集这个列表中的各个元素也是列表，我
们要遍历数据集中的每个元素，一旦发现符合要求的值，则将其添加到新创建的列表中


'''
#计算给定的数据集的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}#数据的创建一个数据字典，它的键值是最后一列的数值
    for featVec in dataSet:
        currentLabel = featVec[-1]#获得标签
        if currentLabel not in labelCounts.keys():
            #如果当前键值不存在，则扩展字典并将当前键值加入字典。
            labelCounts[currentLabel] = 0#标签值是键值
            labelCounts[currentLabel] = +1
            #得到熵之后，我们就可以按照获取最大信息增益的方法划分数据集，下一节我们将具体学习
            #如何划分数据集以及如何度量信息增益。
    shannonEnt = 0.0
    for key in labelCounts:#
        prob = float(labelCounts[key])/numEntries##通过标记数据
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def splitDataSet(dataSet,axis,value):
    '''
    vector as m*n ===> 0~m as m dimension  n as column
    :param dataSet:
    :param axis: it as the column value but the return vector not obtain this column values
    :param value: the value as the shatter point value and return the index of this value axis
    :return:
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
#选择最好的数据集划分方式
'''
第一个要求是，数据必须是一种由列表元素组成的列表，而且所有的列表元素都要具有相同的数据长度；
第二个要求是，数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
数据集一旦满足上述要求，我们就可以在函数的第一行判定当前数据集包含多少特征属性。我们无需限定list中的数据
类型，它们既可以是数字也可以是字符串，并不影响实际计算。
'''
def chooseBestFeatureToSplit(dataSet):
    '''
    we find the best feature split point owe to the max InfoGain
    :param dataSet:
    :return:
    '''
    numFeatures = len(dataSet[0]) - 1
    baseEntroy = calcShannonEnt(dataSet)
    baseInfoGain = 0.0;baseFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntroy - newEntropy
        if(infoGain>baseInfoGain):
            baseInfoGain = infoGain
            baseFeature = i
    return baseFeature

def majorityCnt(classList):
    '''
     Most voting methods determine the classification of the leaf nodes
    :param classList:
    :return:
    '''
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),
      key = operator.itemgetter[1],reverse = True)
    return sortedClassCount[0][0]

def creatTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeast = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeast]
    myTree = {bestFeatLabel:{}}
    del (labels[bestFeast])
    featValues = [example[bestFeast] for example in dataSet]
    uniqueVals = set(featValues) #让数据值唯一的方法--->加set集合
                                 #然后使用Python语言原生的集合（set）数据
                                 #类型。集合数据类型与列表类型相似，不同之处仅在于集合类型中的每个值互不相同。从列表中
                                 #创建集合是Python语言得到列表中唯一元素值的最快方法
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet\
                                        (dataSet,bestFeast,value),subLabels)
    return myTree

'''
    CART
'''
def loadDataSet(fileName):
    dataMat = []
    fr = open(fileName)
    for line in fr.readline():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine)#将每行映射为浮点数
        dataMat.append(fltLine)
    return dataMat

def binSplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:][0]
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:][0]
    return mat0,mat1



import operator
def majortyCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
            classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),
    key = operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

