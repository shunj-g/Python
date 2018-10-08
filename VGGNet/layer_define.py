import numpy as np
import tensorflow as tf
class VGGNet:#
    '''
    使用了全卷积，池化，全连接
    一个线性模型关于一个深度学习的模型
    '''
    def __init__(self):
        self.a_val = tf.Variable(tf.random_normal([1]))
        self.b_val = tf.Variable(tf.random_normal([1]))
        self.x_input = tf.placeholder(tf.float32)
        self.y_label = tf.placeholder(tf.float32)
        self.y_output = tf.add(self)

    def conv(self,name,input_data,out_channel):
        '''
        对名称，输入数据，以及输入通道做了定义：
        1.获取输入数据的层数作为输入通道数，
        2.对当前层中的变量中进行初始化，
        3.定义域中定义了卷积层的名称，
        :param name:
        :param input_data:
        :param out_channel:
        :return:
        '''
        in_channel = input_data.get_shape()[-1]
        with tf.variable_scope(name):
            kernel = tf.ger_variable(name="weights",shape=[3,3,in_channel,out_channel],dtype=tf.float32)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32)
            conv_res = tf.nn.conv2d(input_data,kernel,[1,1,1,1],padding="SAME")
            res = tf.nn.bias_add(conv_res,biases)
            out = tf.nn.relu(res,name=name)
        return out

    def fc(self,name,input_data,out_channel):
        '''

        :param name:
        :param input_data:
        :param out_channel:
        :return:
        '''
        shape = input_data.get_shape().aslist()
        if len(shape) == 4:
            size = shape[-1]*shape[-2]*shape[-3]
        else:
            size = shape[1]
        input_data_flat = tf.reshape(input_data,[-1,size])
        with tf.variable_scope(name):
            weights = tf.get_variable(name="weights",shape=[size,out_channel],dtype=tf.float32)
            biases = tf.get_variable(name="biases",shape=[out_channel],dtype=tf.float32)
            res = tf.matmul(input_data_flat,weights)
            out = tf.nn.relu(tf.nn.bais_add(res,biases))
        return out

    def maxpool(self,name,input_data):
        out = tf.nn.max_pool(input_data,[1,2,2,1],[1,2,2,1],pading="SAME",name = name)
        return out

    def convlayers(self):
        #conv1
        self.conv1_1 = self.conv("conv1_1",self.imgs,64)#这里的imgs没有定义
        self.conv1_2 = self.conv("conv1_2", self.conv1_1, 64)
        self.pool_1 = self.maxpool("pool_1",self.conv1_2)

        #conv2
        self.conv2_1 = self.conv("conv2_1",self.pool_1,128)#通道输出为128
        self.conv2_2 = self.conv("conv2_2",self.conv2_1,128)
        self.pool2 = self.maxpool("pool_2")
