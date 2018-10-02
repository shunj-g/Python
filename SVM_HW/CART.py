from numpy import *

#定义一个CART，用来保存已构成Tree的数据集

def regLeaf(dataSet):  # 返回每个叶子节点的值
    '''
    :param dataSet:
    :return: 返回的值是由dataSet决定的
    '''
    return mean(dataSet[:, -1])##最后一行的数据
    #
def regErr(dataSet):
    return var(dataSet[:, -1]) * shape(dataSet)[0]#数据集的方差*数据集的行数

def binSplitDataSet(dataSet, feature, value):
    '''
    :param dataSet:  数据集;
    :param feature:  数据特征;
    :param value:  切分值;
    :return: mat0, mat1  数据集;
    '''

    mat0 = dataSet[nonzero(dataSet[:, feature] > value)[0], :][0]#matlab中的性质
    mat1 = dataSet[nonzero(dataSet[:, feature] <= value)[0], :][0]
    return mat0, mat1#

def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    '''
    :param dataSet:数据集
    :param leafType:叶子类型
    :param errType:误差类型
    :param ops: 分裂操作阀值(一般为两个数)
    :return cpu:
    '''
    tolS = ops[0]
    tolN = ops[1]
    # 如果所有目标变量都是相同的值:退出和返回值
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:  # 退出条件数1
        return None, leafType(dataSet)
    m, n = shape(dataSet)
    # 最佳特性的选择是由RSS错误从均值减少所驱动的
    S = errType(dataSet)
    bestS = inf####
    bestIndex = 0
    bestValue = 0#
    for featIndex in range(n - 1):
        for splitVal in set(dataSet[:, featIndex]):#得到的数据
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:#如果误差到达最优的情况
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS#得到最佳的误差集
    # 如果下降(S-bestS)小于阈值，就不要分裂
    if (S - bestS) < tolS:#阀值小于1（这里是设定的值）
        return None, leafType(dataSet)  # 退出条件数2
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)#分裂出两个矩阵
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  # 退出条件数3
        return None, leafType(dataSet)###叶子节点的数据类型
    return bestIndex, bestValue  # 返回要分割的最佳特性 # 和用于分割的值

def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):  # 假设数据集是NumPy Mat，那么我们可以数组过滤
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)# 选择最佳分割
    if feat == None: return val  # 如果分裂达到停止条件，返回val
    retTree = {}#定义了一个字典
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree#返回决策树


