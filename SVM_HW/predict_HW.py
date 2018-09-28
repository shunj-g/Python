import numpy
import test
#import csv
import SVM_HW.SVM as SVM
'''
 @author: shunj-g 18/6/23
 SVM只能区分两类数据，要像伸进网络那样去做个总体的平分要调结构
 总体方法是：孤立一个数字，从其他的数字中区分开，然后就可以进行数据识别了
'''

def trainDigits(dataArr,labelArr,kTup=('rbf', 10)):
    # 通过SVM找到了一个分界线的参数值
    b,alphas = SVM.smoP(numpy.mat(dataArr), numpy.mat(labelArr), 200, 0.0001,10000,kTup)
    #print(b,alphas)  得到的是支持向量的参数值
    #得到的数据来进行矩阵化
    datMat = numpy.mat(dataArr)             #获取得到的是标记数据
    labelMat = numpy.mat(labelArr)          #通过数据得到标签
    #数据结构
    svInd = numpy.nonzero(alphas.A>0)[0]#
    print('A>0的数据的索引值的大小为%到',svInd)
    sVs = datMat[svInd]
    labelSV = labelMat[0,svInd]

    print('there are %d Support Vectors' % numpy.shape(sVs)[0])

    m,n = numpy.shape(datMat)
    print(m,n)
    errorCount = 0

    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs,datMat[i,:],kTup)
        predict = kernelEval.T * numpy.multiply(labelSV,alphas[svInd]) + b
        #print('the pridict value is: %d',predict)
        #print('the realizes value is: %d', labelArr[0,i])
        #print(i)
        #print(labelArr[i])
        #print(predict[0,i])
        print(numpy.sign(predict))
        if numpy.sign(predict) != numpy.sign(labelArr):
            errorCount += 1

    print('the training error rate is: %f'% (float(errorCount)/m))

#open函数里的路径根据数据存储的路径来设定\n",
training_data_file = open('mnist/train.csv')
trainning_data_list = training_data_file.readlines()
print(len(trainning_data_list))
training_data_file.close()
#把数据依靠','区分，并分别读入\n",
trainning_list = trainning_data_list[900:1101]
dataArr = []
labelArr =[]
for record in trainning_list:
    all_train_values = record.split(',')
    #print(all_train_values)
    inputs = numpy.sign((numpy.asfarray(all_train_values[1:]))/255.0 * 0.99)#
    # 设置图片与数值的对应关系
    #这里出现了很大的一个错误,,数据的类型不一样
    if int(all_train_values[0]) == 0:###第一个数位上是一个标签数据
        labels = 1
    else:
        labels = -1
    #这里要十分小心，这里是二维数据，在实际操作过程重要始终注意行列的序号
    dataArr.append(inputs)
    labelArr.append(labels)

print(numpy.shape(dataArr)[0],numpy.shape(dataArr)[1])#1000个数字
print(numpy.shape(labelArr)[0])#10000个标签
print(labelArr)
########数据训练########
trainDigits(dataArr,labelArr,kTup=('rbf', 10))
#  K = exp(K/(-1*kTup[1]**2))#k = e^{k/(-1*kTup[1]**2)}  10为参数
########################

###数据的大小是完全可以来进行计算的
'''
def predictDigits(dataArr,labelArr,kTup=('rbf', 10)):
    errorCount = 0
    datMat=numpy.mat(dataArr); labelMat = numpy.mat(labelArr).transpose()
    m,n = numpy.shape(datMat)
    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * numpy.multiply(labelSV,alphas[svInd]) + b
        if numpy.sign(predict)!=numpy.sign(labelArr[i]): errorCount += 1
    print('the test error rate is: %f'% (float(errorCount)/m))
'''


