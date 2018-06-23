from numpy import *
'''
@author shunj-g 18/23/6
'''
#TODO 编写SVM的相关算法，然后就手写体进行识别和分类
'''
#本算法是机遇SMO算法的实现
#从实现上来说，对于标准的SMO能提高速度的地方有：

       1、能用缓存的地方尽量用，例如，缓存核矩阵，减少重复计算，但是增加了空间复杂度；

       2、如果SVM的核为线性核时候，可直接更新，毕竟每次计算的代价较高，于是可以利用旧
       的乘子信息来更新。

       3、关注可以并行的点，用并行方法来改进，例如可以使用MPI，将样本分为若干份，在查
       找最大的乘子时可以现在各个节点先找到局部最大点，然后再从中找到全局最大点；又如
       停止条件是监视对偶间隙，那么可以考虑在每个节点上计算出局部可行间隙，最后在master
       节点上将局部可行间隙累加得到全局可行间隙。
'''

#定义核函数,该函数可以用来进行核函数调整
def kernelTrans(X, A, kTup):
    '''
    #内核或将数据转换为更高维度的空间
    :param X:X的数据是m*n的
    :param A:参数矩阵m*1
    :param kTup: 不同的 kernel function 'lin'---->#linear kernel,,,'rbf'---->#Radial Basis Function kernel
    :return:
    '''
    '''
    总体而言K-->核函数是一个m*m的矩阵  只要K是半正定的就能够作为核函数使用
    '''
    m,n = shape(X)
    K = mat(zeros((m,1)))#K的数据是 m*1的matrix
    if kTup[0]=='lin': K = X * A.T#--->m*n 变成一行了
    elif kTup[0]=='rbf':
        for j in range(m):
            deltaRow = X[j,:] - A#每一行进行运算
            K[j] = deltaRow*deltaRow.T#得到一列的值  得到一个准确值
        # 在NumPy中除是元素级的而不是像Matlab那样的矩阵的除法
        K = exp(K/(-1*kTup[1]**2))
    else: raise NameError('不能得到核函数')
    return K
###
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler,kTup):
        '''
        :param dataMatIn: 数据集----->X
        :param classLabels:类别标签---->Y
        :param C:常数C
        :param toler:容错率---》软间隔SVM
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
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
def calcEk(obj, k):
    '''
    #计算误差
    :param self:
    :param k:
    :return: EK
    '''
    ''' 
        f(x) = w.T*x + b = alpha_i*y_i*x_i.T*x+b  这里的x转化为核函数k(x,x_i)计算出目标函数值
    '''
    #得到的数据的
    fXk = float(multiply(obj.alphas,obj.labelMat.T).T*obj.K[:,k] + obj.b)
    #计算出误差值
    Ek = fXk - float(obj.labelMat[k])
    return Ek
#
def selectJrand(i,m):
    '''
    #要选择任意J中不等于i的随机值
    :param i:
    :param m:
    :return j:选取的J的值
    '''
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j


def selectJ(obj,i,Ei):
    '''
    #使用启发式来进行随机选择j，然后是Ej
    :param obj:
    :param i:
    :param Ei:
    :return:
    '''
    maxK = -1#最大误差率的索引
    maxDeltaE = 0#最大误差率是变化率
    Ej = 0  #最大误差率的
    obj.eCache[i] = [1,Ei]            #设置有效的选择给出最大的E。
    validEcacheList = nonzero(obj.eCache[:,0].A)[0]#返回不等于零的索引号的值
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:     #循环遍历有效的Ecache值，找到能使E最大化的值
            if k == i: continue      #跳过不符合条件
            Ek = calcEk(obj, k)       #计算误差值
            #TODO 这块需要理解Ei是---> Ek是--->
            deltaE = abs(Ei - Ek)     #进行误差值相减得到误差变化率
            if (deltaE > maxDeltaE):  #得到的最大误差变化率
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:                             #在这种情况下(第一次)，我们没有任何有效的eCache值
        j = selectJrand(i, obj.m)
        Ej = calcEk(obj, j)
    return j, Ej                      #它返回第二个的误差

def clipAlpha(aj,H,L):
    '''
    #给alpha上约束值(裁剪除去没有用的alpha值)
    :param aj:
    :param H:
    :param L:
    :return:
    '''
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def updateEk(obj, k):
    '''
    #在任何alpha改变后，更新缓存中的新值
    :param obj:
    :param k:
    :return:
    '''
    Ek = calcEk(obj, k)
    obj.eCache[k] = [1,Ek]

def innerL(i, obj):
    '''

    :param i:
    :param obj:
    :return:
    '''
    Ei = calcEk(obj, i)

    '''
    KTT条件：
            alpha >=0,mu >=0
            yf(X)-1-sigma >=0
            alpha*(y*f(x)-1+sigma) = 0
            sigma >= 0,mu*sigma = 0
            sigma <=1 样本分类正确
    '''
    # 如果目前x不能满足KTT条件
    if (((obj.labelMat[i]*Ei).any() < -obj.toler) and ((obj.alphas[i]).any() < obj.C)) or (((obj.labelMat[i]*Ei).any() > obj.toler) and ((obj.alphas[i]).any() > 0)):
        j,Ej = selectJ(obj, i, Ei)
        alphaIold = obj.alphas[i].copy(); alphaJold = obj.alphas[j].copy()#克隆一个
        if (obj.labelMat[i] != obj.labelMat[j]):
            L = max(0, obj.alphas[j] - obj.alphas[i])
            H = min(obj.C, obj.C + obj.alphas[j] - obj.alphas[i])
        else:
            L = max(0, obj.alphas[j] + obj.alphas[i] - obj.C)
            H = min(obj.C, obj.alphas[j] + obj.alphas[i])
        if L==H: print("L==H"); return 0
        eta = 2.0 * obj.K[i,j] - obj.K[i,i] - obj.K[j,j] #改变为内核
        if eta >= 0: print("eta>=0"); return 0
        obj.alphas[j] -= obj.labelMat[j]*(Ei - Ej)/eta
        obj.alphas[j] = clipAlpha(obj.alphas[j],H,L)
        updateEk(obj, j)
        if (abs(obj.alphas[j] - alphaJold) < 0.00001): print("j not moving enough"); return 0
        obj.alphas[i] += obj.labelMat[j]*obj.labelMat[i]*(alphaJold - obj.alphas[j])#更新i的数量与j相同
        updateEk(obj, i) #这是为Ecache添加的。                 #更新是反方向的
        b1 = obj.b - Ei- obj.labelMat[i]*(obj.alphas[i]-alphaIold)*obj.K[i,i] - obj.labelMat[j]*(obj.alphas[j]-alphaJold)*obj.K[i,j]
        b2 = obj.b - Ej- obj.labelMat[i]*(obj.alphas[i]-alphaIold)*obj.K[i,j]- obj.labelMat[j]*(obj.alphas[j]-alphaJold)*obj.K[j,j]
        if (0 < obj.alphas[i]) and (obj.C > obj.alphas[i]): obj.b = b1
        elif (0 < obj.alphas[j]) and (obj.C > obj.alphas[j]): obj.b = b2
        else: obj.b = (b1 + b2)/2.0
        return 1
    else: return 0

def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):  # Platt SMO
        '''

        :param dataMatIn:
        :param classLabels:
        :param C:
        :param toler:
        :param maxIter:
        :return:
        '''

        obj = optStruct(mat(dataMatIn), mat(classLabels), C, toler,kTup)
        iter = 0
        entireSet = True
        alphaPairsChanged = 0
        while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):
            alphaPairsChanged = 0
            if entireSet:  #
                for i in range(obj.m):
                    alphaPairsChanged += innerL(i, obj)
                    print("fullSet, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            else:  # 越过无界的alphas（拉格朗日乘子）
                nonBoundIs = nonzero((obj.alphas.A > 0) * (obj.alphas.A < C))[0]
                for i in nonBoundIs:
                    alphaPairsChanged += innerL(i, obj)
                    print("non-bound, iter: %d i:%d, pairs changed %d" % (iter, i, alphaPairsChanged))
                iter += 1
            if entireSet:
                entireSet = False  # 整个设置循环切换
            elif (alphaPairsChanged == 0):
                entireSet = True
            print("iteration number: %d" % iter)
        return obj.b, obj.alphas

