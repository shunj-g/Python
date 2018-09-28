'''
使用SVM库来进行手写体识别
'''
from sklearn import *

# open函数里的路径根据数据存储的路径来设定
training_data_file = open('mnist/train.csv')
trainning_data_list = training_data_file.readlines()
print(len(trainning_data_list))
training_data_file.close()

#获得数