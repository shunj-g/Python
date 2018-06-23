from numpy import *
import csv
'''
 @author:shunj-g 18/6/14
'''
#open函数里的路径根据数据存储的路径来设定\n",
training_data_file = open('mnist/train.csv')
trainning_data_list = training_data_file.readlines()
print(len(trainning_data_list))
training_data_file.close()