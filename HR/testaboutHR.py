import numpy
import matplotlib.pyplot
from HR import ANN_BP
'''
 @author:gsj
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
trainning_data_list = trainning_data_list[50:150]
for record in trainning_data_list:
    all_train_values = record.split(',')
    #print(all_train_values)
    inputs = (numpy.asfarray(all_train_values[1:]))/255.0 * 0.99 + 0.01
    #设置图片与数值的对应关系\n",
    targets = numpy.zeros(output_nodes) + 0.001
    targets[int(all_train_values[0])] = 0.99
    net.train(inputs, targets)

test_data_file = open('mnist/test.csv')
test_data_list = test_data_file.readlines()
test_data_file.close()
test_data_list = test_data_list[1:10]

scores = []
for record in test_data_list:
    all_data = record.split(',')
    #correct_number = int(all_values[0])
    #print('该图片对应的数字为:',correct_number)
    #预处理数字图片\n
    inputs = (numpy.asfarray(all_data)) / 255.0 * 0.99 + 0.01
    # 让网络判断图片对应的数字
    outputs = net.query(inputs)
    # 找到数值最大的神经元对应的编号
    label = numpy.argmax(outputs)
    #print('out put reslut is : ', label)
    print('网络认为图片的数字是：', label)
   # if label == correct_number:
   #     scores.append(1)
   # else:
    #    scores.append(0)
#scores_array = asarray(scores)
#print('perfermance = ', scores_array.sum() / scores_array.size)


'''
#%matplotlib inline
#open函数里的路径根据数据存储的路径来设定
data_file = open('/Users/chenyi/Documents/人工智能/mnist_test_10.csv')
data_list =  data_file.readlines()
data_file.close()

#把数据依靠','区分，并分别读入
all_values = data_list[0].split(',')
#第一个值对应的是图片的表示的数字，所以我们读取图片数据时要去掉第一个数值
image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')
'''
