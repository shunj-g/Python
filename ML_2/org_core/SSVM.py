'''
  @author gsj
  SVM算法实现
'''
from numpy import *
import math

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


'''
    简化版的SMO的实现
'''
def smoSample(dataMatIn,classLabels,C,toler,maxIter):#五个参数
    '''
    :param dataMatIn: 数据集----->X
    :param classLabels: 类别标签---->Y
    :param C: 常数C
    :param toler: 容错率
    :param maxIter: 取消前最大的循环次数
    :return:
    '''
    dataMatrix = mat(dataMatIn)#
    labelMat = mat(classLabels).transpose()##-
    b = 0
    m,n = np.shape(dataMatrix)
    alphas = mat(np.zeros((m,1)))
    iter = 0
    while (iter < maxIter):
        alphaPairChanged = 0
        for i in range(m):
            fxi = float(multiply(alphas,labelMat).T*\
                        (dataMatrix*dataMatrix[i,:].T))+b#-->(dataMatrix*dataMatrix[i,:].T)等价于Knn核函数
            Ei = fxi -float(labelMat[i])
            if ((labelMat[i])*Ei< -toler) and (alphas[i] < C) or \
                ((labelMat[i]) * Ei > -toler) and (alphas[i] > 0):
                j = selectJrand(i,m)
                fxj = float(multiply(alphas,labelMat).T*\
                            (dataMatrix*dataMatrix[j,:].T))+b
                Ej = fxj-float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()
                if(labelMat[i] != labelMat[j]):
                    L =  max(0,alphas[j]-alphas[i])
                    H = min(C,C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print("L==H"); continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T-\
                    dataMatrix[i,:]*dataMatrix[i,:].T - \
                    dataMatrix[j,:]*dataMatrix[j,:].T
                if (eta >= 0): print("eta >= 0");continue
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta
                alphas[j] = clipalpha(alphas[j],H,L)
                if(abs(alphas[j] - alphaJold) < 0.00001):\
                        print("J is not moving enough");continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i] - alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[i,:].T-\
                     labelMat[j]*(alphas[j]-alphaJold)*\
                     dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i] - alphaIold)*\
                     dataMatrix[i,:]*dataMatrix[j,:].T-\
                     labelMat[j]*(alphas[j]-alphaJold)*\
                     dataMatrix[j,:]*dataMatrix[j,:].T
                if(0 < alphas[i])and(C > alphas[i]): b = b1
                elif (0 < alphas[i])and(C > alphas[i]):b = b2
                else:b = (b1 +b2)/2.0
                alphaPairChanged += 1
                print("iter:  %d i:%d, pairs changed %d" %(iter,i,alphaPairChanged))
        if(alphaPairChanged == 0):iter += 1
        else:iter = 0
        print("iteration number: %d" % iter)
    return b,alphas


