import numpy
import scipy.special
import sys
'''
@author:gsj
'''

sys.setrecursionlimit(1000000) #括号中的值为递归深度
'''
Scipy是一个高级的科学计算库，它和Numpy联系很密切，
Scipy一般都是操控Numpy数组来进行科学计算，所以可以说是基于Numpy之上了。
Scipy有很多子模块可以应对不同的应用，例如插值运算，优化算法、图像处理、数学统计等
scipy.cluster 	向量量化
scipy.constants 	数学常量
scipy.fftpack 	快速傅里叶变换
scipy.integrate 	积分
scipy.interpolate 	插值
scipy.io 	数据输入输出
scipy.linalg 	线性代数
scipy.ndimage 	N维图像
scipy.odr 	正交距离回归
scipy.optimize 	优化算法
scipy.signal 	信号处理
scipy.sparse 	稀疏矩阵
scipy.spatial 	空间数据结构和算法
scipy.special 	特殊数学函数
scipy.stats 	统计函数
'''
class NeuralNetWork:
    def __init__(self,iNodeNum,hNodeNum,oNodeNum,LearningRate):#初始化网络的输入层，中间层，输出层
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

    def train(self,input_list,output_list):#根据输入的训练数据更新相应的节点的链路权重
       #将数据集转换我我们需要的格式numpy的二维格式
        dataSet = numpy.array(input_list,ndmin = 2).T
        labelSet = numpy.array(output_list,ndmin = 2).T
       # 计算信号经过输入层后产生的信号量
        hide_inputs = numpy.dot(self.wih,dataSet)
       # 中间层神经元对输入的信号做激活函数后得到输出信号
        hide_outputs = self.activation_function(hide_inputs)
       # 输出层接收来自中间层的信号量
        outputs = numpy.dot(self.who,hide_outputs)
       #输出层对信号量进行激活函数后得到最终输出信号
        final_outputs = self.activation_function(outputs)

       #误差计算与更新(误差从后往前更新)
        output_errors = labelSet-final_outputs
        hide_errors = numpy.dot(self.who.T,output_errors)
       # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        A = output_errors * final_outputs * (1 - final_outputs)
        self.who += self.lr*numpy.dot(A, numpy.transpose(hide_outputs))
        B = hide_inputs*(1 - hide_inputs)
        C = hide_errors * B
        self.wih += self.lr*numpy.dot(C,numpy.transpose(dataSet))
        pass
    def query(self,inputdata):#根据输入数据进行计算并得到答案
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




