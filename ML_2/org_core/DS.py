'''
   @author  18/
   单层决策树
   ADBOOST
'''
from numpy import  *
'''

'''
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    '''
    # dataMatirx：要分类的数据
    # dimen：维度
    # threshVal：阈值
    # threshIneq：有两种，‘lt’=lower than，‘gt’=greater than
    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threshIneq:
    :return:
    '''
    retArray = ones((shape(dataMatrix)[0],1))#shape(dataMatrix)[0]获取行值
    if threshIneq == 'lt':#初始化的时候全部是+1，，通过判定来获得相应的决策值
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
'''
    构建单层决策树
    算法：  将最小错误率minError设为+∞
            对数据集中的每一个特征（第一层循环）：
               对每个步长（第二层循环）：
                   对每个不等号（第三层循环）：
                        建立一棵单层决策树并利用加权数据集对它进行测试
                        如果错误率低于minError，则将当前单层决策树设为最佳单层决策树
            返回最佳单层决策树
    
'''
def buildStump(dataArr,classLabels,D):#构建单层决策树
    '''
    # dataArr: 数据集
    # classLabels：标签
    # D：由每个样本的权重构成的矩阵
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    '''
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n  = shape(dataMatrix) #m为数据个数，n为每条数据含有的样本数（也就是特征）
    numSteps = 10.0;bestStrump = {};bestClasEst = mat(zeros((m,1)))# 初始化最好的分类器为[[0],[0],[0],...]
    minError = inf#将最小错误率minError设为+∞
    for i in range(n):
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1): #对每个步长
            for inequal in ['lt','gt']:# 每个条件，大于阈值是1还是小于阈值是1
                threshVal = (rangeMin+float(j)*stepSize)# 阈值设为最小值+第j个步长
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)# 将dataMatrix的第i个特征inequal阈值的置为1，否则为-1
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] == 0# 预测对的置0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStrump['dim'] = i###存储键值对
                    bestStrump['thresh'] = threshVal
                    bestStrump['ineq'] = inequal
    return bestStrump,minError,bestClasEst
'''
        @author gsj   18/
        构建adpboost 算法
        算法：
        对每次迭代：
                利用buildStump()函数找到最佳的单层决策树
                将最佳单层决策树加入到单层决策树数组
                计算alpha
                计算新的权重向量D
                更新累计类别估计值
                如果错误率等于0.0，则退出循环---->通过错误率来进行优化
'''
def adaBoostTrainDS(dataArr,classLabels,numIt = 40):
    weakClassArr = []#弱分类
    m = shape(dataArr)[0]#获取行向量的行数
    D = mat(ones((m,1))/m)#初始化权重向量，给每个样本相同的权重，[[1/m],[1/m],[1/m],...]
    aggClassEst = mat(zeros((m,1)))# 初始化每个样本的预估值为0
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D) # 构建一棵单层决策树，返回最好的树，错误率和分类结果
        print("D:",D.T)
        alpha = float(0.5*log((1.0 - error)/max(error,1e-16)))#计算alpha#计算分类器权重
        bestStump['alpha'] = alpha#将alpha值也加入最佳树字典
        weakClassArr.append(bestStump)# 将XX加入弱分类器数组
        print("classEst:",classEst.T)
        #更新权重向量D
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))#
        D = D/D.sum()
        # 累加错误率，直到错误率为0或者到达迭代次数
        aggClassEst += alpha*classEst#累加错误
        print("aggCladdEst:",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst)!=mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m ##平均错误率
        aggClassEst += alpha * classEst
        print("total error:", errorRate, "\n")
        if errorRate == 0.0: break;
    return weakClassArr



