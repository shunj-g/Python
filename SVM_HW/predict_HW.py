import numpy
import csv
'''
 @author:shunj-g 18/6/23
'''
#open函数里的路径根据数据存储的路径来设定\n",
training_data_file = open('mnist/train.csv')
trainning_data_list = training_data_file.readlines()
print(len(trainning_data_list))
training_data_file.close()

#把数据依靠','区分，并分别读入\n",
trainning_list = trainning_data_list[1:42001]
for record in trainning_list:
    all_train_values = record.split(',')
    #print(all_train_values)
    inputs = (numpy.asfarray(all_train_values[1:]))/255.0 * 0.99 + 0.01
    #设置图片与数值的对应关系\n",
