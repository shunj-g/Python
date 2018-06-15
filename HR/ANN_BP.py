import numpy
import scipy.special
'''
 @author:shunj-g 18/6/14
'''

class NeuralNetWork:
    def __init__(self,iNodeNum,hNodeNum,oNodeNum,LearningRate):
        '''
        #初始化网络的输入层，中间层，输出层，在本识别中，
        #是以28*28一张图片的像素点作为总的输入点，通过不断的训练最后
        #可以将各个参数调到最优
        :param iNodeNum:输入层节点数
        :param hNodeNum:中间层节点数
        :param oNodeNum:输出层节点数
        :param LearningRate:
        '''
        self.input = iNodeNum
        self.hide = hNodeNum
        self.output = oNodeNum
        self.lr = LearningRate#学习率
        #初始化权重，有两个权重矩阵，一个是wih表示输入层和中间层节点间链路权重形成的矩阵\n",
        #一个是who,表示中间层和输出层间链路权重形成的矩阵
        self.wih = numpy.random.rand(self.hide,self.input) - 0.5
        self.who = numpy.random.rand(self.output,self.hide) - 0.5
        #定义激活函数
        self.activation_function = lambda x : scipy.special.expit(x)
        pass

    def train(self,input_list,output_list):
        '''
        #根据输入的训练数据更新相应的节点的链路权重
        :param input_list:
        :param output_list:
        :return:
        '''
       #将数据集转换我我们需要的格式numpy的二维格式
        dataSet = numpy.array(input_list,ndmin = 2).T
        labelSet = numpy.array(output_list,ndmin = 2).T
       # 信号经过输入层后产生的信号量

        hidden_inputs = numpy.dot(self.wih, dataSet)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 输出层接收来自中间层的信号量
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(final_inputs)

        # 计算误差
        output_errors = labelSet - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        numpy.transpose(dataSet))
        pass
    def query(self,inputdata):
        '''
        #根据输入数据进行计算并得到答案
        :param inputdata:
        :return:final_outputs
        '''
        # 计算信号经过输入层后产生的信号量
        hide_inputs = numpy.dot(self.wih, inputdata)
        # 中间层神经元对输入的信号做激活函数后得到输出信号
        hide_outputs = self.activation_function(hide_inputs)
        # 输出层接收来自中间层的信号量
        outputs = numpy.dot(self.who, hide_outputs)
        # 输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(outputs)
        return final_outputs
        pass




