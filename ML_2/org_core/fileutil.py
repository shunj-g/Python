'''
  @author gsj    18/
  数据格式转化
'''
#########################
import numpy as np
import random
'''
    txt ----> martrix
'''
def file2martrix(filename):
    fr = open(filename)#读取文件
    arrayLines = fr.readlines()#读出数据
    numberOfLines = len(arrayLines)#获取行数
    returnMat = np.zeros((numberOfLines,3))#一定是内部的矩阵
    classLabelVector = []
    index = 0
    for line in arrayLines:
        line = line.strip()#首先使用函数line.strip()截取掉所有的回车字符
        listFormLine = line.split('\t')#使用'\t'进行分解
        returnMat[index,:] = listFormLine[0:3]
        '''
         0, 1, 2, 3
        -4,-3,-2,-1
       '''
        classLabelVector.append(int(listFormLine[-1]))#-
        index += 1
    return returnMat,classLabelVector

def loadDataSet(filename):
    dataMat = [];  labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])#存储数据
        labelMat.append(float(lineArr[2]))#存储标记
    return dataMat,labelMat
def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j

def clipalpha(aj,H,L):
    if aj > H:
        aj = H
    if aj < L:
       aj = L
    return aj