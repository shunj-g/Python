import numpy
#open函数里的路径根据数据存储的路径来设定\n",
training_data_file = open('mnist/train.csv')
trainning_data_list = training_data_file.readlines()
print(len(trainning_data_list))
training_data_file.close()
#把数据依靠','区分，并分别读入\n",
trainning_list = trainning_data_list[1:1001]
dataArr = []
labelArr = []
for record in trainning_list:
    all_train_values = record.split(',')
    #print(all_train_values)
    inputs = numpy.sign((numpy.asfarray(all_train_values[1:]))/255.0 * 0.99)#
    # 设置图片与数值的对应关系
    print('当前的标签值为',all_train_values[0])
    if int(all_train_values[0]) == 3:###第一个数位上是一个标签数据
        labels = 1
        print('true')
    else:
        print('false')
        labels = -1
    #这里要十分小心，这里是二维数据，在实际操作过程重要始终注意行列的序号
    dataArr.append(inputs)
    labelArr.append(labels)

print(numpy.shape(dataArr)[0],numpy.shape(dataArr)[1])#1000个数字
print(numpy.shape(labelArr)[0])#1000个标签