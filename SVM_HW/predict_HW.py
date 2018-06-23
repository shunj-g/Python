import numpy
import csv

import SVM_HW.SVM as SVM
'''
 @author:shunj-g 18/6/23
 SVM只能区分两类数据，要像伸进网络那样去做个总体的平分要调结构
'''

def trainDigits(dataArr,labelArr,kTup=('rbf', 10)):
    b,alphas = SVM.smoP(dataArr, labelArr, 200, 0.0001, 10000,kTup)
    datMat=numpy.mat(dataArr); labelMat = numpy.mat(labelArr).transpose()
    svInd=numpy.nonzero(alphas.A>0)[0]
    sVs=datMat[svInd]
    labelSV = labelMat[svInd]
    print ('there are %d Support Vectors' % numpy.shape(sVs)[0])

    m,n = numpy.shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = SVM.kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * numpy.multiply(labelSV,alphas[svInd]) + b
        if numpy.sign(predict)!=numpy.sign(labelArr[i]): errorCount += 1
    print('the training error rate is: %f'% (float(errorCount)/m))
#open函数里的路径根据数据存储的路径来设定\n",
training_data_file = open('mnist/train.csv')
trainning_data_list = training_data_file.readlines()
print(len(trainning_data_list))
training_data_file.close()

#把数据依靠','区分，并分别读入\n",
trainning_list = trainning_data_list[1:101]
dataArr = []
labelArr = []
for record in trainning_list:
    all_train_values = record.split(',')
    #print(all_train_values)
    inputs = (numpy.asfarray(all_train_values[1:]))/255.0 * 0.99 + 0.01
    # 设置图片与数值的对应关系",
    labels = numpy.zeros(10) + 0.001 #0~9十个数
    labels[int(all_train_values[0])] = 0.999
    dataArr.append(inputs)
    labelArr.append(labels)

#print(numpy.shape(dataArr)[0],numpy.shape(dataArr)[1])
#print(numpy.shape(labelArr)[0],numpy.shape(labelArr)[1])

trainDigits(dataArr,labelArr,kTup=('rbf', 10))

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


