import numpy
import matplotlib.pyplot
import csv
from HR import ANN_BP
'''
 @author:shunj-g 18/6/14
'''
#初始化网络
input_nodes = 784#(28*28)
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.3
net = ANN_BP.NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#读入训练数据\n",
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
    targets = numpy.zeros(output_nodes) + 0.001
    targets[int(all_train_values[0])] = 0.99
    net.train(inputs, targets)

test_data_file = open('mnist/test.csv')
test_data_list = test_data_file.readlines()
print(len(test_data_list))
test_data_file.close()
test_data_list = test_data_list[1:28001]
#test_data_list = trainning_data_list[30001:30011]
scores = []
with open('mnist/sample_submission.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    #县写入columns_name
    writer.writerow(['ImageId','Label'])
    #写入多行
    count = 1
    for record in test_data_list:
        all_data = record.split(',')
        inputs = (numpy.asfarray(all_data)) / 255.0 * 0.99 + 0.01
        # 让网络判断图片对应的数字
        outputs = net.query(inputs)
        # 找到数值最大的神经元对应的编号
        label = numpy.argmax(outputs)
        writer.writerow([count, label])
        count  = count+1

'''
for record in test_data_list:
    all_data = record.split(',')
    print('应该输出的数字是：', all_data[0])
    inputs = (numpy.asfarray(all_data[1:])) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = net.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    #print('out put reslut is : ', label)
    print('网络认为图片的数字是：', label)
'''
