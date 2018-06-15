from numpy import *
'''
@author shunj-g 18/14/6
'''
#TODO 编写SVM的相关算法，然后就手写体进行识别和分类

#定义核函数,该函数可以用来进行核函数调整
def kernelTrans(X, A, kTup):
    '''
    #内核或将数据转换为更高维度的空间
    :param X:X的数据是m*n的
    :param A:参数矩阵m*1
    :param kTup: 不同的 kernel function 'lin'---->#linear kernel,,,'rbf'---->#Radial Basis Function kernel
    :return:
    '''
    m,n = shape(X)
    K = mat(zeros((m,1)))#K的数据是 m*1的matrix
    if kTup[0]=='lin': K = X * A.T#--->m*n
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A
            K[j] = deltaRow*deltaRow.T#得到一列的值
        # 在NumPy中除是元素级的而不是像Matlab那样的矩阵的除法
        K = exp(K/(-1*kTup[1]**2))
    else: raise NameError('不能得到核函数')
    return K
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        '''

        :param dataMatIn: 数据集----->X
        :param classLabels:类别标签---->Y
        :param C:常数C
        :param toler:容错率
        :param kTup: kernel 的类型
        '''
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.toler = toler
        self.m = shape(dataMatIn)[0]#这里的shape(matrix) 返回的是一个参数list,0-->行数，1-->列数
        self.alphas = mat(zeros((self.m,1)))#m*1的matrix
        self.b = 0
        self.eCache = mat(zeros((self.m, 2))) #第一列是作为有效标志
        self.K = mat(zeros((self.m,self.m)))#K是m*m的值
        for i in range(self.m):#K的值是一个完全不同的
            self.K[:,i] = kernelTrans(self.X, self.X[:,i], kTup)



